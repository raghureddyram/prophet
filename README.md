
This project was built with python 3.11 and poetry.

INSTALLATION:

please ensure `pip3 install poetry` succeeds.

`poetry install`

`touch .env`

(in the .env file add your api key for Open ai. should be `OPENAI_API_KEY="your_key_here"`)

USAGE"

`poetry run earnings-reviewer --year=2022`

or

`poetry run earnings-reviewer --year=2023`

or

`poetry run earnings-server`
navigate to http://localhost:8000/earnings/{year}/answers?question={your_question}


Please ask a question in a complete sentence, with proper punctuation. The program expects something like "How much revenue did Microsoft make this year?" or "Who is Satya Nadella?" or "What lines of business does Microsoft have?"

![alt text](https://github.com/raghureddyram/prophet/blob/master/example.png?raw=true)



--------
HIGH LEVEL

This project follows the Retrieval Augmented Generation pattern. In a nutshell, the user picks a year in which Microsoft has reported their annual earnings, which corresponds to a pdf of a 10k. Then the pdf data is pre-processed by extracting text and stripping unnecessary newline characters. Cleaned data is seperated into a list of chunks of equal token size. These chunks are then turned into embeddings (vectors), which can then be indexed for retrieval.
Next a user asks a question. The question is then turned into an embedding, which will then be compared to the top k similar chunks of context in our vector index. Given a top k set of context embeddings and a question embedding, the call to the decoder llm can be made. The decoder llm first receives system instructions, followed by a question, and finally by k most relevent chunks of context. Each chunk of context deserves its own set of messages to be passed to the decoder. After all of the context is in, the decoder llm is instructed to analyze and respond with their assessment of the question.

---------
CONSIDERATIONS:

In order to build something quick and to avoid the rountrips for sending large chunks of data over the wire unnecessarily, I decided to use Meta's FAISS library, since I know that under the hood many of the top vector DB companies today are using FAISS to power their vector indexing algorithms. I could have setup a pinecone db instance, but then that would require storing ENV keys in github, which seemed like a security vulnerability. This way the user only has to have an openai account to get up and running, and doesn't have to wait for a docker image to build and connect all the things.

I also wanted to avoid the compute overhead of re-indexing the vector index at runtime. This is why I didn't allow the user to switch between year 2022 and 2023 midway and instead forced them to pick the year as an argument to the program. I could have pre-computed the indexes for each year, saved to pickles, then loaded directly from pickles - but then I wouldn't be able to demonstrate my understanding of how/why we would typically persist data to a vector index.

I wanted to avoid unnecessary development costs. The open source encoder model performs relatively well locally and allowed me to avoid having to pay openai more per token for their encoder usage. If I had more time then I would have explored hosting an open-source decoder llm like say mistral on huggingface or aws, then experimented with prompting said llm. I was already somewhat familiar with open-ai's api, so to speed up my development cycle I just went with what I could get up and running.

Getting deterministic, idempotent output from open-ai's llm is a huge struggle. I've learned that prompts can only get you so far, so testing for deterministic output seems like its an ongoing challenge. For this reason I would again want to host an open-source model that I can then fine-tune to at least fulfil my testing needs.

Token size considerations were a big struggle. Initially I was sending too much context at a time, which kept causing issues. If the max_token_size was too little, then there wasn't usually enough context to find useful answers to most of my test questions. I settled on a token size of 1000, and top 6 nearest neighbors, just to prove the point that if at least 6000 tokens are to be sent then the program should still work, even if the max token limit is around 4000. 

