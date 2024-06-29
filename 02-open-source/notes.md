# 02 Open Source LLMs Notes 

### I'm using my local GPU

Install libraries 

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install -U transformers accelerate bitsandbytes sentencepiece
```

### Huggingface Hub and Environment variables set up 

1. Create a read token in Huggingface website
2. Login into the HF with CLI: `Login command will ask for your HF token`
    ```bash
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login
    huggingface-cli whoami
    ```

    or you can login inside of the notebook:
    ```python
    from huggingface_hub import login
    login(token='YOUR_TOKEN')
    ```
3. Set cache directory:
    ```python
    import os; os.environ['HF_HOME'] = '/SSD/slava/huggingface'
    ```

### Using LLM from HF in our RAG flow

Let's use [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1). Go to the link and accept the agreement.

Load the model:
```python
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

Then, modify our `def llm(prompt)` function from week-01.

```python
def llm(prompt):
    response = generator(prompt, max_length=500, temperature=0.7, top_p=0.95, num_return_sequences=1)
    response_final = response[0]['generated_text']
    return response_final[len(prompt):].strip()
```

### Ollama: Running LLM without GPU

**Ollama** is a package that runs LLMs locally.

Running Ollama: 

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama start
ollama run phi3 
```

Access the LLM using OpenAI API: 

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)
```

### Ollama + Elastic Search with docker-compose

To run both Ollama and Elastic Search, so we need **docker-compose.yaml** file to run multi-container applications.  See [docker-compose.yaml](./docker-compose.yaml)

```bash
docker-compose up
```

Then, we can use the *RAG* notebook from week-01 by changing client in `def llm(prompt)` function. 


### Streamlit UI

Instead of running our application on the jupyter notebook, we can create friendly UI using **streamlit**. 

```bash
pip install streamlit
```

See the [documentation](https://docs.streamlit.io/) to create more complex UI, but a simple chat where we have input box and an "ask" button is created as follows: 

```python
# Input box
user_input = st.text_input("Enter your input:")

# Button to invoke the RAG function
if st.button("Ask"):
    with st.spinner('Processing...'):
        # Call the RAG function
        output = rag(user_input)
        st.success("Completed!")
        st.write(output)
```

To run the application: 

```bash
streamlit run qa_faq.py
```