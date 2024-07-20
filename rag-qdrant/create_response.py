import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)
gpt_model_name = "gpt-4o-mini"
# Pricing per million tokens
input_price_per_million_tokens = 0.150   # $0.150 per 1 million tokens
output_price_per_million_tokens = 0.600  # $0.600 per 1 million tokens

def estimate_gpt_cost(input_text, output_text):
    # Function to count tokens, assuming a simple whitespace tokenization for demonstration purposes
    def count_tokens(text):
        return len(text.split())
    
    # Calculate the number of tokens in input and output
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text)
    
    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * input_price_per_million_tokens
    output_cost = (output_tokens / 1_000_000) * output_price_per_million_tokens
    
    # Total cost
    total_cost = input_cost + output_cost
    
    return total_cost

def generate_full_prompt(query, context):
    prompt = f"""
    You are an expert assistant. Use only the information from the provided context to answer the question accurately and comprehensively.
    Context:
    {context}

    Question:
    {query}

    Please provide a detailed and clear answer strictly based on the context provided, without relying on any external knowledge or pre-existing information.
    """
    return prompt

def create_response(query, context):
    prompt = generate_full_prompt(query, context)

    response = client.chat.completions.create(
        model=gpt_model_name,
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ], 
        temperature=0.0
    )

    answer = response.choices[0].message.content.strip()
    call_cost = estimate_gpt_cost(prompt, answer)

    return prompt, answer, call_cost

def main():
    # Example usage
    query = "What are the benefits of using RAG systems?"
    context = """
    Retrieval-Augmented Generation (RAG) systems combine the strengths of information retrieval and generation models. They are useful for tasks where leveraging external knowledge is crucial. Benefits include improved accuracy, the ability to handle large knowledge bases, and dynamic updating of information without retraining the model. RAG systems provide contextually relevant information from vast external sources, enhancing the quality and relevance of generated responses.
    """

    _, answer, cost = create_response(query, context)
    print("Answer:", answer)
    print(f"Estimated Cost: {cost}$")

if __name__ == "__main__":
    main()
