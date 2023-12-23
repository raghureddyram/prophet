from openai import OpenAI
import tiktoken

class AnswerGenerator():
    BASE_PROMPT = f"""You are a financial analyst. You are going to be looking at the Microsoft 10k for 2022. A client is asking questions about Microsoft's performance. Please answer to the best of your ability, only AFTER seeing the 'ALL_PARTS_SENT' indicator.

        Your answers are correct, high-quality, and written by an domain expert. 

        User question: {{}}

        Contexts:
        {{}}
        """

    def __init__(self, encoder_model, index, paragraphs, relevant_docs_count=6):
        self.encoder_model = encoder_model
        self.decoder_model = OpenAI()
        self.index = index 
        self.paragraphs = paragraphs
        self.top_k = relevant_docs_count
        self.decoder_model_name = "gpt-3.5-turbo"

    def generate_answer(self, question="What number is represented here by 168 Billion?"):
        query_embedding = self.encoder_model.encode([question])
        distances, neighbors = self.index.search(query_embedding, self.top_k)

        # provide the top k nearest neighbors as context
        relevant_context = [f"{i}. {self.paragraphs[self.index]}" for i, self.index in enumerate(neighbors[0])]
        responses = []
        messages = [
            {"role": "system", "content": self.BASE_PROMPT},
            {
                "role": "user",
                "content": f"QUESTION: {question}. Do not give an answer yet.",
            },
        ]
        tokenizer = tiktoken.encoding_for_model('gpt-4')
       

        # Append least accurate to most accurate, popping least accurate if we hit token length
        for i in range(len(relevant_context)-1, -1, -1):
            chunk = relevant_context[i]
            messages.append({"role": "user", "content": f"context: {chunk}. Do not give an answer yet."})

            # Check if total tokens exceed the model's limit and remove least relevant chunk. Do not remove system context or question.
            while (sum(len(tokenizer.encode(msg["content"])) for msg in messages) > 4090):
                messages.pop(2)

            response = self.decoder_model.chat.completions.create(model=self.decoder_model_name, messages=messages)
            chatgpt_response = response.choices[0].message.content.strip()
            responses.append(chatgpt_response)

        messages.append({"role": "user", "content": "context: {context}. ALL PARTS SENT. Please analze the entire context and give the answer."})
        response = self.decoder_model.chat.completions.create(model=self.decoder_model_name, messages=messages)
        final_response = response.choices[0].message.content.strip()
        responses.append(final_response)
        print(responses[-1])
        print("----------")
        return responses[-1]