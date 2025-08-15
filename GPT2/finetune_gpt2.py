from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import wandb  # Import Weights & Biases
import logging  # Enhanced logging
from tqdm.auto import tqdm  # Progress bars

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def finetune_gpt2(model_name_or_path, train_data_file, output_dir):
    # Initialize wandb
    wandb.init(
        project="gpt2-finetune-mental-health",
        config={
            "model": model_name_or_path,
            "dataset": train_data_file,
            "epochs": 1,
            "batch_size": 32
        }
    )
    logger.info("Weights & Biases initialized")

    # Load GPT-2 Model and Tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    logger.info(f"Loaded model and tokenizer: {model_name_or_path}")

    # Load training dataset
    logger.info(f"Loading dataset from: {train_data_file}")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_data_file,
        block_size=128,
    )

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Set training arguments with wandb integration
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',  # Log directory for tensorboard
        logging_steps=100,  # Log metrics every 100 steps
        report_to="wandb",  # Enable wandb reporting
    )

    # Custom callback for tqdm progress bars and logging
    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_local_process_zero:
                logger.info(f"Step {state.global_step} - Loss: {logs['loss']:.4f}")

        def on_train_begin(self, args, state, control, **kwargs):
            if state.is_local_process_zero:
                self.progress_bar = tqdm(total=state.max_steps, desc="Training")

        def on_step_end(self, args, state, control, **kwargs):
            if state.is_local_process_zero:
                self.progress_bar.update(1)

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_local_process_zero:
                self.progress_bar.close()

    # Train model with enhanced progress tracking
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback]  # Add custom callback
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning completed!")

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to: {output_dir}")

    # Finish wandb run
    wandb.finish()
    logger.info("Weights & Biases session closed")


if __name__ == "__main__":
    model_name_or_path = "gpt2"
    train_data_file = "./mental_health_data.txt"
    output_dir = "./output"

    # Ensure wandb is installed (uncomment if needed)
    # import sys
    # !{sys.executable} -m pip install wandb

    finetune_gpt2(model_name_or_path, train_data_file, output_dir)