"""
Minesweeper Model Template
Teams should modify this file to load their fine-tuned model
"""
import time
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class MinesweeperAgent(object):
    def __init__(self, **kwargs):
        # ✅ CHANGE 1: Update model path to your merged model
        model_name = "/workspace/my_minesweeper_model_merged_latest"
        
        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> tuple:
        """
        Generate LLM response for Minesweeper action.

        Args:
            message: Game state prompt(s)
            system_prompt: System prompt for the model
            **kwargs: Generation parameters

        Returns:
            (response, token_count, generation_time)
        """
        # ✅ CHANGE 2: Match training system prompt
        if system_prompt is None:
            system_prompt = "You are a Minesweeper AI. Output ONLY valid JSON. No explanations, no reasoning, just {\"type\":\"reveal\",\"row\":N,\"col\":N}."

        if isinstance(message, str):
            message = [message]

        # Prepare all messages for batch processing
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        # Convert all messages to text format
        texts = []
        for messages in all_messages:
            # ✅ REMOVED enable_thinking (not in notebook)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Tokenize all texts together with padding
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)

        # Conduct batch text completion
        if tgps_show_var:
            start_time = time.time()

        # ✅ CHANGE 3: Add temperature, top_p, do_sample (CRITICAL!)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 128),  # Keep 128 (rule requirement)
            temperature=kwargs.get("temperature", 0.7),         # ✅ ADDED
            top_p=kwargs.get("top_p", 0.9),                     # ✅ ADDED
            do_sample=True,                                      # ✅ ADDED
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        if tgps_show_var:
            generation_time = time.time() - start_time

        # Decode only the generated tokens, skipping the prompt
        batch_outs = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Clean up outputs
        batch_outs = [output.strip() for output in batch_outs]
        print(batch_outs)
        
        # Calculate token count if needed
        if tgps_show_var:
            token_len = sum(len(generated_ids[i]) - model_inputs.input_ids.shape[1]
                          for i in range(len(generated_ids)))

        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )

        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


if __name__ == "__main__":
    # Test the model
    agent = MinesweeperAgent()

    test_prompt = """6x6 5mines
? ? ? ? ? ?
? ? ? ? ? ?
? ? ? ? ? ?
? ? ? ? ? ?
? ? ? ? ? ?
? ? ? ? ? ?
Reply ONLY {"type":"reveal","row":N,"col":N} or {"type":"flag","row":N,"col":N}
Example: {"type":"reveal","row":2,"col":3}
DO NOT explain. Just the JSON."""

    response, tl, tm = agent.generate_response(
        test_prompt,
        tgps_show=True,
        max_new_tokens=128,
    )

    print("Model response:")
    print(response)
    if tl and tm:
        print(f"\nTokens: {tl}, Time: {tm:.2f}s, TGPS: {tl/tm:.2f}")
