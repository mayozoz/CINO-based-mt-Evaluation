import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Optional
 
class TibZhBERTScorer:
    def __init__(self, model_path: str = "CINOBERTScore/hfl/cino-large-v2"):
        """
        Args:
            model_path: Path to the CINO model
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load CINO
        print("Loading CINO model...")
        try:
            # First try with AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
        except Exception as e:
            print(f"AutoTokenizer failed with: {e}")
            print("Trying with XLMRobertaTokenizer instead...")
            from transformers import XLMRobertaTokenizer, XLMRobertaModel
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
            self.model = XLMRobertaModel.from_pretrained(model_path)
    
        # Move models to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        print("Models loaded successfully!")


    def score_tib_to_zh(
        self, 
        references: List[str], 
        hypotheses: List[str], 
        batch_size: int = 8
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute BERTScore for Tibetan→Chinese translations
    
        Args:
            references: Reference Chinese translations
            hypotheses: Generated Chinese translations
            batch_size: Batch size for processing
    
        Returns:
            Tuple of (precision_scores, recall_scores, f1_scores)
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have the same length")
    
        P_scores, R_scores, F_scores = [], [], []
    
        # Process in batches
        for i in tqdm(range(0, len(references), batch_size), desc="Computing BERTScore"):
        # for i in range(0, len(references), batch_size):
            batch_refs = references[i:i+batch_size]
            batch_hyps = hypotheses[i:i+batch_size]
    
            # Get Chinese embeddings for both references and hypotheses
            ref_embeddings, ref_masks = self._get_chinese_embeddings(batch_refs)
            hyp_embeddings, hyp_masks = self._get_chinese_embeddings(batch_hyps)
    
            # Compute BERTScore for this batch
            batch_P, batch_R, batch_F = self._compute_bertscore_batch(
                ref_embeddings, ref_masks, hyp_embeddings, hyp_masks
            )
    
            P_scores.extend(batch_P)
            R_scores.extend(batch_R)
            F_scores.extend(batch_F)
    
        return P_scores, R_scores, F_scores
    

    def score_ref_free_tib_to_zh(
        self, 
        sources: List[str], 
        translations: List[str], 
        batch_size: int = 8
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute BERTScore for Tibetan→Chinese translations
        Uses reference free cross-lingual scoring
    
        Args:
            sources: Reference Chinese translations
            translations: Generated Chinese translations
            batch_size: Batch size for processing
    
        Returns:
            Tuple of (precision_scores, recall_scores, f1_scores)
        """

        if len(sources) != len(translations):
            raise ValueError("Sources and translations must have the same length")
        
        P_scores, R_scores, F_scores = [], [], []

        for i in tqdm(range(0, len(sources), batch_size), desc="Computing BERTScore"):
        # for i in range(0, len(references), batch_size):
            batch_srcs = sources[i:i+batch_size]
            batch_hyps = translations[i:i+batch_size]
    
            # Get Chinese embeddings for both references and hypotheses
            src_embeddings, src_masks = self._get_tibetan_embeddings(batch_srcs)
            tgt_embeddings, tgt_masks = self._get_chinese_embeddings(batch_hyps)
    
            # Compute BERTScore for this batch
            batch_P, batch_R, batch_F = self._compute_cross_lingual_bertscore_batch(
                src_embeddings, src_masks, tgt_embeddings, tgt_masks
            )
    
            P_scores.extend(batch_P)
            R_scores.extend(batch_R)
            F_scores.extend(batch_F)

        return P_scores, R_scores, F_scores

    def score_zh_to_tib(
        self, 
        references: List[str], 
        hypotheses: List[str], 
        batch_size: int = 8
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute BERTScore for Chinese→Tibetan translations
    
        Args:
            references: Reference Tibetan translations
            hypotheses: Generated Tibetan translations
            batch_size: Batch size for processing
    
        Returns:
            Tuple of (precision_scores, recall_scores, f1_scores)
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have the same length")
    
        P_scores, R_scores, F_scores = [], [], []
    
        # Process in batches
        for i in tqdm(range(0, len(references), batch_size), desc="Computing BERTScore"):
        # for i in range(0, len(references), batch_size):
            batch_refs = references[i:i+batch_size]
            batch_hyps = hypotheses[i:i+batch_size]
    
            # Get Tibetan embeddings for both references and hypotheses
            ref_embeddings, ref_masks = self._get_tibetan_embeddings(batch_refs)
            hyp_embeddings, hyp_masks = self._get_tibetan_embeddings(batch_hyps)
    
            # Compute BERTScore for this batch
            batch_P, batch_R, batch_F = self._compute_bertscore_batch(
                ref_embeddings, ref_masks, hyp_embeddings, hyp_masks
            )
    
            P_scores.extend(batch_P)
            R_scores.extend(batch_R)
            F_scores.extend(batch_F)
    
        return P_scores, R_scores, F_scores
    

    def score_ref_free_zh_to_tib(
        self, 
        sources: List[str], 
        translations: List[str], 
        batch_size: int = 8
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute BERTScore for Tibetan→Chinese translations
        Uses reference free cross-lingual scoring
    
        Args:
            sources: Reference Chinese translations
            translations: Generated Chinese translations
            batch_size: Batch size for processing
    
        Returns:
            Tuple of (precision_scores, recall_scores, f1_scores)
        """

        if len(sources) != len(translations):
            raise ValueError("Sources and translations must have the same length")
        
        P_scores, R_scores, F_scores = [], [], []

        for i in tqdm(range(0, len(sources), batch_size), desc="Computing BERTScore"):
        # for i in range(0, len(references), batch_size):
            batch_srcs = sources[i:i+batch_size]
            batch_hyps = translations[i:i+batch_size]
    
            # Get Chinese embeddings for both references and hypotheses
            src_embeddings, src_masks = self._get_chinese_embeddings(batch_srcs)
            tgt_embeddings, tgt_masks = self._get_tibetan_embeddings(batch_hyps)
    
            # Compute BERTScore for this batch
            batch_P, batch_R, batch_F = self._compute_cross_lingual_bertscore_batch(
                src_embeddings, src_masks, tgt_embeddings, tgt_masks
            )
    
            P_scores.extend(batch_P)
            R_scores.extend(batch_R)
            F_scores.extend(batch_F)

        return P_scores, R_scores, F_scores


    def _get_tibetan_embeddings(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for Tibetan texts using TiBERT"""
        # Preprocess Tibetan texts
        processed_texts = [self.preprocess_tibetan(text) for text in texts]
    
        # Tokenize
        inputs = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
    
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
    
        return embeddings, inputs['attention_mask']
    
    def _get_chinese_embeddings(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for Chinese texts using CINO."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
    
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
    
        return embeddings, inputs['attention_mask']
    
    def _compute_bertscore_batch(
        self,
        ref_embeddings: torch.Tensor,
        ref_masks: torch.Tensor,
        hyp_embeddings: torch.Tensor,
        hyp_masks: torch.Tensor
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute BERTScore for a batch of examples
    
        Args:
            ref_embeddings: Reference embeddings (batch_size, ref_len, hidden_size)
            ref_masks: Reference attention masks (batch_size, ref_len)
            hyp_embeddings: Hypothesis embeddings (batch_size, hyp_len, hidden_size)
            hyp_masks: Hypothesis attention masks (batch_size, hyp_len)
    
        Returns:
            Lists of precision, recall, and F1 scores
        """
        batch_size = ref_embeddings.size(0)
        P_scores, R_scores, F_scores = [], [], []
    
        for i in range(batch_size):
            # Get valid (non-padded) tokens
            ref_valid_mask = ref_masks[i].bool()
            hyp_valid_mask = hyp_masks[i].bool()
    
            # Extract valid embeddings (excluding [CLS], [SEP], and padding)
            ref_valid_embeddings = ref_embeddings[i][ref_valid_mask][1:-1]  # Remove [CLS] and [SEP]
            hyp_valid_embeddings = hyp_embeddings[i][hyp_valid_mask][1:-1]  # Remove [CLS] and [SEP]
    
            # Skip if either sequence is empty after removing special tokens
            if ref_valid_embeddings.size(0) == 0 or hyp_valid_embeddings.size(0) == 0:
                P_scores.append(0.0)
                R_scores.append(0.0)
                F_scores.append(0.0)
                continue
    
            # Normalize embeddings for cosine similarity
            ref_norm = F.normalize(ref_valid_embeddings, p=2, dim=1)
            hyp_norm = F.normalize(hyp_valid_embeddings, p=2, dim=1)
    
            # Compute similarity matrix: (ref_len, hyp_len)
            sim_matrix = torch.mm(ref_norm, hyp_norm.t())
    
            # Precision: For each hypothesis token, find max similarity with reference
            precision = sim_matrix.max(dim=0)[0].mean().item()
    
            # Recall: For each reference token, find max similarity with hypothesis
            recall = sim_matrix.max(dim=1)[0].mean().item()
    
            # F1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
            P_scores.append(precision)
            R_scores.append(recall)
            F_scores.append(f1)
    
        return P_scores, R_scores, F_scores
    

    def _compute_cross_lingual_bertscore_batch(
            self,
            src_embeddings: torch.Tensor,
            src_masks: torch.Tensor,
            tgt_embeddings: torch.Tensor,
            tgt_masks: torch.Tensor 
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute BERTScore for a batch of examples
        Used for reference free evaluation where source and target are in different languages
    
        Args:
            src_embeddings: Reference embeddings (batch_size, src_len, hidden_size)
            src_mask: Reference attention masks (batch_size, src_len)
            tgt_embeddings: Hypothesis embeddings (batch_size, tgt_len, hidden_size)
            tgt_mask: Hypothesis attention masks (batch_size, tgt_len)
    
        Returns:
            Lists of precision, recall, and F1 scores
        """

        batch_size = src_embeddings.size(0)
        P_scores, R_scores, F_scores = [], [], []

        for i in range(batch_size):
            # Get valid (non-padded) tokens
            src_valid_mask = src_masks[i].bool()
            tgt_valid_mask = tgt_masks[i].bool()
            # Extract valid embeddings (excluding [CLS], [SEP], and padding)
            src_valid_embeddings = src_embeddings[i][src_valid_mask][1:-1]  # Remove [CLS] and [SEP]
            tgt_valid_embeddings = tgt_embeddings[i][tgt_valid_mask][1:-1]  # Remove [CLS] and [SEP]
    
            # Skip if either sequence is empty after removing special tokens
            if src_valid_embeddings.size(0) == 0 or tgt_valid_embeddings.size(0) == 0:
                P_scores.append(0.0)
                R_scores.append(0.0)
                F_scores.append(0.0)
                continue

            # For cross-lingual comparison, we need to project embeddings to a common space
            # Since we're using different models, we'll use cosine similarity directly
            # but with a normalization factor to account for different embedding spaces

            src_norm = F.normalize(src_valid_embeddings, p=2, dim=1)
            tgt_norm = F.normalize(tgt_valid_embeddings, p=2, dim=1)
    
            sim_matrix = torch.mm(src_norm, tgt_norm.t())
            
            # Apply a scaling factor for cross-lingual comparison
            # This helps account for the fact that cross-lingual similarities are typically lower
            sim_matrix = sim_matrix * 1.2 # Empirical scaling factor
            sim_matrix = torch.clamp(sim_matrix, -1, 1) # keep in range

    
            # Precision: For each token, find max similarity with source
            precision = sim_matrix.max(dim=0)[0].mean().item()
    
            # Recall: For each token, find max similarity with source
            recall = sim_matrix.max(dim=1)[0].mean().item()
    
            # F1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
            P_scores.append(precision)
            R_scores.append(recall)
            F_scores.append(f1)
    
        return P_scores, R_scores, F_scores
    

    def preprocess_tibetan(self, text: str) -> str:
        """
        Clean and normalize Tibetan text
        Note: This is a simplified version. For better results, use proper Tibetan NLP tools.
        """
        try:
            # Try to use pybo if available
            from pybo import WordTokenizer
            tok = WordTokenizer()
            words = tok.tokenize(text)
            return " ".join([w.text for w in words if w.pos != 'punct'])
        except ImportError:
            # Fallback: basic cleaning
            import re
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            # Remove common punctuation (adapt as needed)
            text = re.sub(r'[།༎༏༐༑༔]', '', text)
            return text
    
    def evaluate_dataset(
        self,
        references: List[str],
        hypotheses: List[str],
        direction: str = "tib2zh",
        batch_size: int = 8
    ) -> dict:
        """
        Evaluate a complete dataset and return summary statistics
    
        Args:
            references: List of reference translations
            hypotheses: List of hypothesis translations
            direction: "tib2zh" or "zh2tib"
            batch_size: Batch size for processing
    
        Returns:
            Dictionary with evaluation metrics
        """
        if direction == "tib2zh":
            P, R, F = self.score_tib_to_zh(references, hypotheses, batch_size)
        elif direction == "zh2tib":
            P, R, F = self.score_zh_to_tib(references, hypotheses, batch_size)
        else:
            raise ValueError("Direction must be 'tib2zh' or 'zh2tib'")
    
        results = {
            "precision": {
                "mean": np.mean(P),
                "std": np.std(P),
                "scores": P
            },
            "recall": {
                "mean": np.mean(R),
                "std": np.std(R),
                "scores": R
            },
            "f1": {
                "mean": np.mean(F),
                "std": np.std(F),
                "scores": F
            },
            "num_examples": len(references)
        }
    
        return results


    def evaluate_ref_free_dataset(
        self,
        sources: List[str],
        translations: List[str],
        direction: str = "tib2zh",
        batch_size: int = 8
    ) -> dict:
        """
        Evaluate a complete dataset and return summary statistics
        Used for reference free cross-lingual scoring
    
        Args:
            sources: List of reference translations
            translations: List of hypothesis translations
            direction: "tib2zh" or "zh2tib"
            batch_size: Batch size for processing
    
        Returns:
            Dictionary with evaluation metrics
        """
        if direction == "tib2zh":
            print(type(sources), type(translations), type(batch_size))
            P, R, F = self.score_ref_free_tib_to_zh(sources, translations, batch_size)
        elif direction == "zh2tib":
            P, R, F = self.score_ref_free_zh_to_tib(sources, translations, batch_size)
        else:
            raise ValueError("Direction must be 'tib2zh' or 'zh2tib'")
    
        results = {
            "precision": {
                "mean": np.mean(P),
                "std": np.std(P),
                "scores": P
            },
            "recall": {
                "mean": np.mean(R),
                "std": np.std(R),
                "scores": R
            },
            "f1": {
                "mean": np.mean(F),
                "std": np.std(F),
                "scores": F
            },
            "num_examples": len(sources)
        }
    
        return results
    

def example_usage():

    # Initialize scorer
    scorer = TibZhBERTScorer()
    
    # Example data (you would replace with your actual data)
    # Tibetan to Chinese
    chinese_references = [
        "你好，世界！",
        "我的名字是丹增。",
        "今天天气很好。"
    ]
    
    chinese_hypotheses = [
        "你好世界！",  # Missing comma
        "我叫丹增。",    # Different phrasing
        "今天的天气很好。"  # Extra particle
    ]
    
    # Evaluate Tibetan→Chinese translations
    # print("Evaluating Tibetan→Chinese translations...")
    results_tib2zh = scorer.evaluate_dataset(
        chinese_references, 
        chinese_hypotheses, 
        direction="tib2zh",
        batch_size=2
    )
    
    print(f"Average F1 Score: {results_tib2zh['f1']['mean']:.4f} ± {results_tib2zh['f1']['std']:.4f}")
    print(f"Average Precision: {results_tib2zh['precision']['mean']:.4f}")
    print(f"Average Recall: {results_tib2zh['recall']['mean']:.4f}")
    
    # Example for Chinese→Tibetan (you would need Tibetan reference data)
    tibetan_references = [
        "བཀྲ་ཤིས་བདེ་ལེགས།",
        "ངའི་མིང་ལ་བསྟན་འཛིན་ཟེར།"
    ]
    
    tibetan_hypotheses = [
        "བཀྲ་ཤིས་བདེ་ལེགས།",
        "ང་གི་མིང་ལ་བསྟན་འཛིན་ཟེར།"
    ]
    
    # print("\nEvaluating Chinese→Tibetan translations...")
    results_zh2tib = scorer.evaluate_dataset(
        tibetan_references,
        tibetan_hypotheses,
        direction="zh2tib",
        batch_size=2
    )
    
    # print(f"Average F1 Score: {results_zh2tib['f1']['mean']:.4f} ± {results_zh2tib['f1']['std']:.4f}")