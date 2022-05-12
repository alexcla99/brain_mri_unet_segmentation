# Parameters of the dataset
dataset_params = {
    "data_type": ".jpg",
	"img_size": (256, 256),
	"val_samples": 500
}

# Parameters of the model
model_params = {
	"num_classes": 3,
	"batch_size": 16,
	"filters": [32, 64, 128, 256, 512],
	"num_epochs": 5 # TODO 100
}