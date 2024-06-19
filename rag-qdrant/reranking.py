import copy
from flashrank.Ranker import Ranker, RerankRequest

def reranking(query, passages, choice):
    # Create a deep copy of passages to prevent modification of the original list
    passages_copy = copy.deepcopy(passages)

    if choice == "ms-marco-TinyBERT-L-2-v2":
        ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    elif choice == "ms-marco-MiniLM-L-12-v2":
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    else:
        print("Did not select valid model")
        return []

    rerankrequest = RerankRequest(query=query, passages=passages_copy)
    reranked_passages = ranker.rerank(rerankrequest)

    return reranked_passages


def main() -> None:
    query = "What is an LLM?"
    choice = "Nano"
    passages = [
    {
        "id":1,
        "text":"The use of lookahead decoding in LLMs is a step towards optimizing machine learning models. While not specifically addressing what an LLM is, it does delve into methods to enhance model performance."
    },
    {
        "id":2,
        "text":"Language models, or LLMs, have seen significant advancements in efficiency and effectiveness. Projects like vLLM highlight how critical these innovations are for reducing costs and improving performance in real-world applications."
    },
    {
        "id":3,
        "text":"LLMs, or large language models, are designed to understand and generate human-like text based on the training data they have been fed. These models are integral to numerous applications, from chatbots to advanced analytical tools."
    },
    {
        "id":4,
        "text":"An LLM, or Large Language Model, is a type of artificial intelligence that processes and generates language based on massive datasets. They are used in various applications to enhance interaction and automation through natural language understanding."
    },
    {
        "id":5,
        "text":"A Large Language Model (LLM) is a sophisticated AI technology designed to understand and produce human-like text by analyzing extensive collections of written language. These models are the backbone of many modern AI applications, enabling more intuitive and interactive user experiences."
    },
    {
        "id":6,
        "text":"The stock market's recent volatility can be attributed to a mixture of economic indicators and investor sentiment, not directly related to technological advancements in AI or any specific models like LLMs."
    },
    {
        "id":7,
        "text":"In the world of sports, using analytics has become crucial for improving team performance and scouting opponents. This approach differs significantly from the computational techniques used in language modeling."
    },
    {
        "id":8,
        "text":"Cooking methods such as sous-vide have revolutionized how professional chefs and home cooks prepare meals, focusing on temperature control and timing rather than AI or language processing."
    },
    {
        "id":9,
        "text":"Environmental conservation efforts are crucial in combating climate change. Initiatives like reforestation and wildlife protection are essential, distinct from the computational or digital solutions provided by LLMs."
    },
    {
        "id":10,
        "text":"The historical impact of the Renaissance on modern Western art cannot be understated, showcasing a period of significant cultural rebirth which contrasts with the technological nature of language models and AI advancements."
    }
    ]
    print(f"Original indexes order: {[passage['id'] for passage in passages]}")
    reranked_result = reranking(query, passages, choice)
    print(f"Reranker indexes order: {[passage['id'] for passage in reranked_result]}") 


if __name__ == "__main__":
    main()