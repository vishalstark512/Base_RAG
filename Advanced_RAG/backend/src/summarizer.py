from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src.utils import setup_logging
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram
import time
import torch

nltk.download('punkt', quiet=True)
logger = setup_logging()

# Prometheus metrics
SUMMARY_COUNTER = Counter('summaries_generated_total', 'Total number of summaries generated')
SUMMARY_ERRORS = Counter('summarization_errors_total', 'Total number of summarization errors')
SUMMARY_LATENCY = Histogram('summarization_latency_seconds', 'Summarization latency in seconds')

class Summarizer:
    def __init__(self, config: Dict[str, Any]):
        self.method = config.get('method', 'abstractive')
        self.model_name = config.get('model_name', 'facebook/bart-large-cnn')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.method == 'abstractive':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=self.device)
        elif self.method == 'extractive':
            self.summarizer = pipeline("summarization", model=self.model_name, device=self.device)
        else:
            raise ValueError(f"Unsupported summarization method: {self.method}")
        
        self.max_length = config.get('max_length', 150)
        self.min_length = config.get('min_length', 50)
        self.do_sample = config.get('do_sample', False)
        self.num_beams = config.get('num_beams', 4)
        
        logger.info(f"Summarizer initialized with method: {self.method}, model: {self.model_name}")

    def summarize(self, documents: List[Dict[str, Any]], query_type: str) -> str:
        start_time = time.time()
        try:
            combined_text = self._combine_documents(documents)
            
            if self.method == 'abstractive':
                summary = self._abstractive_summarize(combined_text)
            elif self.method == 'extractive':
                summary = self._extractive_summarize(combined_text)
            
            logger.info(f"Generated {self.method} summary for query type: {query_type}")
            SUMMARY_COUNTER.inc()
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}", exc_info=True)
            SUMMARY_ERRORS.inc()
            return "Unable to generate summary due to an error."
        finally:
            SUMMARY_LATENCY.observe(time.time() - start_time)

    def summarize_with_citations(self, documents: List[Dict[str, Any]], query_type: str) -> Dict[str, Any]:
        summary = self.summarize(documents, query_type)
        citations = self._generate_citations(summary, documents)
        return {
            "summary": summary,
            "citations": citations
        }

    def _generate_citations(self, summary: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for idx, doc in enumerate(documents):
            if doc['content'] in summary:
                start = summary.index(doc['content'])
                end = start + len(doc['content'])
                citations.append({
                    "start": start,
                    "end": end,
                    "document_id": doc['id']
                })
        return citations

    def _combine_documents(self, documents: List[Dict[str, Any]], max_tokens: int = 1024) -> str:
        combined = ""
        token_count = 0
        for doc in documents:
            content = doc['content']
            tokens = self.tokenizer.tokenize(content)
            if token_count + len(tokens) > max_tokens:
                remaining = max_tokens - token_count
                combined += self.tokenizer.convert_tokens_to_string(tokens[:remaining]) + "..."
                break
            combined += content + " "
            token_count += len(tokens)
        return combined.strip()

    def _abstractive_summarize(self, text: str) -> str:
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt', truncation=True).to(self.device)
        summary_ids = self.model.generate(
            inputs['input_ids'], 
            num_beams=self.num_beams, 
            max_length=self.max_length, 
            min_length=self.min_length, 
            do_sample=self.do_sample
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def _extractive_summarize(self, text: str) -> str:
        sentences = sent_tokenize(text)
        num_sentences = max(1, min(len(sentences), self.max_length // 10))  # Rough estimate
        summary = self.summarizer(text, max_length=self.max_length, min_length=self.min_length, num_sentences=num_sentences)
        return ' '.join(summary)

    def multi_document_summarize(self, documents: List[Dict[str, Any]], query_type: str) -> str:
        combined_text = ' '.join([doc['content'] for doc in documents])
        
        if self.method == 'abstractive':
            return self._abstractive_summarize(combined_text)
        elif self.method == 'extractive':
            return self._extractive_summarize(combined_text)

    def query_focused_summarize(self, documents: List[Dict[str, Any]], query: str, query_type: str) -> str:
        combined_text = ' '.join([doc['content'] for doc in documents])
        
        if self.method == 'abstractive':
            prompt = f"Summarize the following text in the context of this query: '{query}'\n\n{combined_text}"
            return self._abstractive_summarize(prompt)
        elif self.method == 'extractive':
            sentences = sent_tokenize(combined_text)
            query_terms = set(query.lower().split())
            relevant_sentences = [sent for sent in sentences if any(term in sent.lower() for term in query_terms)]
            return ' '.join(relevant_sentences[:self.max_length // 10])

if __name__ == "__main__":
    # Test the Summarizer
    config = {
        'method': 'abstractive',
        'model_name': 'facebook/bart-large-cnn',
        'max_length': 150,
        'min_length': 50,
        'do_sample': False,
        'num_beams': 4
    }
    summarizer = Summarizer(config)
    test_docs = [
        {'id': '1', 'content': 'Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals.'},
        {'id': '2', 'content': 'AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.'}
    ]
    summary = summarizer.summarize(test_docs, 'informational')
    print("Generated summary:")
    print(summary)

    summary_with_citations = summarizer.summarize_with_citations(test_docs, 'informational')
    print("\nSummary with citations:")
    print(summary_with_citations)

    multi_doc_summary = summarizer.multi_document_summarize(test_docs, 'informational')
    print("\nMulti-document summary:")
    print(multi_doc_summary)

    query_focused_summary = summarizer.query_focused_summarize(test_docs, "What is the goal of AI?", 'informational')
    print("\nQuery-focused summary:")
    print(query_focused_summary)