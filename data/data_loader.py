from data.dataset import load_dataloader
from configs.config import get_config

# Load the configuration
config = get_config()


def partition_data_loader(partition_id):
    """
    Load data for a specific partition based on the configuration settings.

    Args:
        partition_id (int): Identifier for the data partition.

    Returns:
        tuple: Train, test, and validation data loaders, along with the number of examples.
    """
    train_loader, test_loader, val_loader, num_examples = load_dataloader(
        partition_id=partition_id,
        n_partitions=config['client'],
        batch_size=config['data']['batch_size'],
        scale=config['data']['scale'],
        use_smote=config['data']['smote'],
        use_rus=config['data']['rus'],
        encode=config['data']['encode'],
        n_pca_components=config['data']['pca']
    )

    return train_loader, test_loader, val_loader, num_examples
