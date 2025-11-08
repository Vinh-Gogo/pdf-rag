from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import os
import gc
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class LLMInstructCorrector:
    """
    ✅ Vietnamese text correction using instruction-tuned LLMs (optimized for speed)
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
        """
        Initialize model + tokenizer with optimized GPU settings.
        """
        # ⚙️ GPU performance tuning
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🚀 Loading model: {model_name} ...")
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set padding side to 'left' for decoder-only models
        self.tokenizer.padding_side = 'left'
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Model loaded and ready!\n")

    def _load_model(self):
        """Load and optimize the model for inference."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()

        # Only compile for large GPUs (10GB+)
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory
            if mem > 10 * 1024**3:
                print("⚙️ Compiling model for optimized inference...")
                model = torch.compile(model)
        return model

    def _calculate_max_tokens(self, text: str) -> int:
        """Limit token generation to reduce latency."""
        length = len(text.split())
        return min(max(length, 512), 1024)

    def correct_batch(self, texts: list) -> list:
        """Batch correction for faster processing."""
        messages = [
            [
                {"role": "system", "content": (
                    "You are an expert proofreader. Only correct spelling, "
                    "grammar, and punctuation in Vietnamese. Preserve names, "
                    "formatting, and do not summarize or translate."
                )},
                {"role": "user", "content": f"\n\n{text}"}
            ]
            for text in texts
        ]

        inps = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        inputs = self.tokenizer(inps, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results = []
        for i in range(len(texts)):
            gen = out_ids[i][inputs.input_ids.shape[1]:]
            results.append(self.tokenizer.decode(gen, skip_special_tokens=True).strip())

        # Clean memory (important for long jobs)
        del inputs, out_ids
        torch.cuda.empty_cache()
        gc.collect()

        return results

    def read_all_txt_list(self, folder: str = "../data/contents") -> list:
        """Read all text files."""
        files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".txt")],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        pages = [open(os.path.join(folder, f), encoding="utf-8").read() for f in files]
        return pages

    def process_and_save(self, output_folder: str = "../data/results/testing"):
        """
        Batch + threaded text correction for speed.
        """
        os.makedirs(output_folder, exist_ok=True)
        pages = self.read_all_txt_list()

        batch_size = 4 if torch.cuda.is_available() else 1  # tune for GPU size
        total = len(pages)
        print(f"📄 Found {total} pages → Processing in batches of {batch_size}...\n")

        # Threading for I/O parallelism
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for i in range(0, total, batch_size):
                batch = pages[i:i + batch_size]
                futures.append(executor.submit(self._process_batch, batch, i, output_folder))

            for f in as_completed(futures):
                idx = f.result()
                print(f"✅ Batch starting at page {idx + 1} completed")

        print("\n🎯 All pages corrected and saved successfully!")

    def _process_batch(self, batch, start_idx, output_folder):
        """Helper for threaded batch correction."""
        corrected_batch = self.correct_batch(batch)
        for j, text in enumerate(corrected_batch):
            page_num = start_idx + j + 1
            out_path = os.path.join(output_folder, f"corrected_page_{page_num}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        return start_idx


if __name__ == "__main__":
    corrector = LLMInstructCorrector()
    corrector.process_and_save()
