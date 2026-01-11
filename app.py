import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="EmpathyBot", page_icon="🤗")

st.title("🤗 EmpathyBot: Your AI Emotional Support Buddy")
st.write("I'm here to listen. Tell me what's on your mind.")

# --- 2. LOAD PRE-TRAINED ML MODEL ---
# We use a caching decorator so the model loads only once (preventing slow reloads)
@st.cache_resource
def load_model():
    # 'microsoft/DialoGPT-medium' is a robust conversational model trained on 147M conversation turns.
    # It acts as the "brain" of the bot.
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# --- 3. INITIALIZE CHAT HISTORY ---
# Streamlit refreshes the code on every interaction, so we use 'session_state' to remember the chat.
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcoming system message
    st.session_state.messages.append({"role": "assistant", "content": "Hi there. I'm listening. How are you feeling today?"})

# --- 4. DISPLAY CHAT INTERFACE ---
# Loop through history and display messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. PROCESS USER INPUT & GENERATE RESPONSE ---
if user_input := st.chat_input("Type your message here..."):
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # B. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Encode user input + history
            # We encode the new input and append the eos_token (End of Sentence)
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

            # 2. Append to chat history tensor (for context awareness)
            # If we have history, append new input to it. If not, start fresh.
            if "chat_history_ids" in st.session_state:
                bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
            else:
                bot_input_ids = new_user_input_ids

            # 3. Generate response
            # max_length=1000: Limits response size
            # pad_token_id: Handles padding
            chat_history_ids = model.generate(
                bot_input_ids, 
                max_length=1000, 
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Save the raw history ids to session state for the next turn
            st.session_state.chat_history_ids = chat_history_ids

            # 4. Decode response
            # We take only the tokens generated *after* the input (the new response)
            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            st.markdown(response)
    
    # C. Save Bot Response to History
    st.session_state.messages.append({"role": "assistant", "content": response})