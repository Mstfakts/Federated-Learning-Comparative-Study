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
        scale=False,  # Already scaled
        use_smote=config['data']['smote'],
        use_rus=config['data']['rus'],
        encode=False,  # Already encoded
        n_pca_components=config['data']['pca'],
        kbest=config['data']['kbest']
    )

    return train_loader, test_loader, val_loader, num_examples
