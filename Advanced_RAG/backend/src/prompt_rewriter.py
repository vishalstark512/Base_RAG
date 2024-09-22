from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils import setup_logging
from typing import List, Union
from prometheus_client import Counter, Histogram
import time
import torch
from functools import lru_cache

logger = setup_logging()

# Prometheus metrics
REWRITE_COUNTER = Counter('prompt_rewrites_total', 'Total number of prompt rewrites')
REWRITE_ERRORS = Counter('prompt_rewrite_errors_total', 'Total number of prompt rewrite errors')
REWRITE_LATENCY = Histogram('prompt_rewrite_latency_seconds', 'Prompt rewrite latency in seconds')

class PromptRewriter:
    def __init__(self, model_name: str = "t5-base"):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"PromptRewriter initialized with model: {model_name} on device: {self.device}")
        except Exception as e:
            logger.error(f"Error initializing PromptRewriter: {str(e)}", exc_info=True)
            raise

    @lru_cache(maxsize=1000)
    def rewrite(self, original_prompt: str, max_length: int = 100) -> List[str]:
        start_time = time.time()
        try:
            input_text = f"rewrite query: {original_prompt}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=3,
                num_beams=5,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2
            )

            rewritten_prompts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            logger.info(f"Original prompt: '{original_prompt}' rewritten to {len(rewritten_prompts)} variations")
            REWRITE_COUNTER.inc()
            return rewritten_prompts
        except Exception as e:
            logger.error(f"Error in prompt rewriting: {str(e)}", exc_info=True)
            REWRITE_ERRORS.inc()
            return [original_prompt]  # Return the original prompt if rewriting fails
        finally:
            REWRITE_LATENCY.observe(time.time() - start_time)

    def expand(self, original_prompt: str, max_length: int = 150) -> str:
        start_time = time.time()
        try:
            input_text = f"expand query: {original_prompt}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                num_beams=3,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )

            expanded_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Original prompt: '{original_prompt}' expanded to: '{expanded_prompt}'")
            REWRITE_COUNTER.inc()
            return expanded_prompt
        except Exception as e:
            logger.error(f"Error in prompt expansion: {str(e)}", exc_info=True)
            REWRITE_ERRORS.inc()
            return original_prompt  # Return the original prompt if expansion fails
        finally:
            REWRITE_LATENCY.observe(time.time() - start_time)

    def clear_cache(self):
        """Clear the LRU cache for the rewrite method."""
        self.rewrite.cache_clear()
        logger.info("PromptRewriter cache cleared")

if __name__ == "__main__":
    # Test the PromptRewriter
    rewriter = PromptRewriter()
    
    original_prompt = "What is the capital of France?"
    rewritten_prompts = rewriter.rewrite(original_prompt)
    print("Rewritten prompts:")
    for prompt in rewritten_prompts:
        print(f"- {prompt}")
    
    expanded_prompt = rewriter.expand(original_prompt)
    print(f"\nExpanded prompt: {expanded_prompt}")
    
    cached_rewrite = rewriter.rewrite(original_prompt)
    print(f"\nCached rewrite (should be faster): {cached_rewrite[0]}")
    
    rewriter.clear_cache()