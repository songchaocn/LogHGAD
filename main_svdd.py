from torch.optim.lr_scheduler import ReduceLROnPlateau

from CONSTANTS import *

from graph_construction import Log2Hypergraph
from util import logger, cut_by_613
import metrics
from svdd import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='BGL')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--model_name', type=str, default='graph_svdd')
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--window_size', default=50, type=int)
    parser.add_argument('--step_size', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    args = parser.parse_args()
    return args


def evaluate(model, hypergraphs, criterion, template_embedding, dataset, model_name, threshold=0.5):
    model.eval()
    y_true = []
    y_proba = []
    val_loss = 0.0
    with torch.no_grad():
        for hypergraph in tqdm(hypergraphs):
            x = template_embedding.get_embeddings(hypergraph.x)
            hyper_edge_index = hypergraph.hyper_edge_index
            probabilities = model(x, hyper_edge_index)
            val_loss += criterion(probabilities, torch.tensor([hypergraph.y], dtype=torch.float)).item()
            y_true.append(hypergraph.y)
            y_proba.append(probabilities)

    result = metrics.evaluate(y_true, y_proba, threshold=threshold)
    logger.info(f"Evaluation {dataset} in {model_name}: " +
                ", ".join(["%s=%s" % (k, v) for k, v in result.items()]))
    return result['f1_score'], val_loss / len(hypergraphs)


if __name__ == "__main__":
    logger = logger('main_svdd')
    args = parse_args()
    dataset = args.dataset
    model_name = args.model_name
    mode = args.mode
    hidden_size = args.hidden_size
    window_size = args.window_size
    step_size = args.step_size
    batch_size = args.batch_size

    num_epochs = 30

    log2Hypergraph = Log2Hypergraph(dataset, window_size=window_size, step_size=step_size)
    hypergraphs = log2Hypergraph.get_hypergraphs()
    template_embedding = log2Hypergraph.get_template_embedding()

    del log2Hypergraph
    gc.collect()

    train, dev, test = cut_by_613(hypergraphs)

    logger.info(
        f'Model: {model_name},mode: {mode},model hyperparameters: number of epochs: {num_epochs}, batch size: {batch_size}')
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    output_model_dir = os.path.join(save_dir, f'models/{model_name}/{dataset}/model')

    last_model_output = output_model_dir + f'/hidden={hidden_size}_window={window_size}_epoch={num_epochs}_last.pt'
    best_model_output = output_model_dir + f'/hidden={hidden_size}_window={window_size}_epoch={num_epochs}_best.pt'

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    criterion = HybridSVDDLoss(nu=0.1, alpha=0.5)

    if mode == 'train':
        input_size = 300
        output_size = 128
        logger.info(
            f"Input_size:{input_size}, Hidden_size:{hidden_size}, Window_size:{window_size}, Step_size:{step_size}")
        start_time = time.time()
        encoder = HypergraphEncoder()
        model = HyperGraphDeepSVDD(encoder)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        logger.info("Initializing center")
        model.init_center(train, template_embedding)
        logger.info(f"Center initialized")
        best_f1 = 0.0
        for epoch in range(num_epochs):
            random.shuffle(train)
            model.train()

            start = time.strftime("%H:%M:%S")
            logger.info(
                f"Starting  epoch: {epoch + 1} | phase: train | start time: {start} | learning rate: {optimizer.param_groups[0]['lr']}")
            total_loss = 0
            count = 0
            distances = []
            labels = []
            for data in tqdm(train, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                data = data
                distance = model(template_embedding.get_embeddings(data.x), data.hyper_edge_index)
                if count < batch_size:
                    distances.append(distance)
                    labels.append(data.y)
                    count += 1
                    continue
                else:
                    loss = criterion(torch.stack(distances, dim=0).flatten(),
                                     torch.tensor(labels, dtype=torch.float))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    total_loss += loss.item()
                    count = 0
                    distances.clear()
                    labels.clear()
            logger.info('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss))
            if dev:
                logger.info('Testing on validation set.')
                threshold = compute_threshold(model, dev, template_embedding, quantile=0.95)
                f1_score, val_loss = evaluate(model, dev, criterion, template_embedding, dataset, model_name,
                                              threshold)
                scheduler.step(val_loss)
                if f1_score > best_f1:
                    best_f1 = f1_score
                    logger.info("Exceed best f: history = %.2f, current = %.2f" % (best_f1, f1_score))
                    torch.save(model.state_dict(), best_model_output)
                    logger.info('Best model saved.')
            elapsed_time = time.time() - start_time
            logger.info('elapsed_time: {:.3f}s'.format(elapsed_time))
            torch.save(model.state_dict(), last_model_output)
        logger.info('Finished Training')
    elif mode == 'test':
        model = HyperGraphDeepSVDD(HypergraphEncoder())
        model.load_state_dict(torch.load(best_model_output))
        logger.info('Starting to init center...')
        model.init_center(train, template_embedding)
        logger.info('========================Starting testing best model========================')
        threshold = compute_threshold(model, test, template_embedding, quantile=0.95)
        logger.info(f'Threshold: {threshold}')
        evaluate(model, test, criterion, template_embedding, dataset, model_name, threshold)
        logger.info('Finished Testing')
        model.load_state_dict(torch.load(last_model_output))
        logger.info('========================Starting testing last model========================')
        evaluate(model, test, criterion, template_embedding, dataset, model_name, threshold)
        logger.info('Finished Testing')
    else:
        logger.error('Mode %s is not supported yet.' % mode)
        raise NotImplementedError
