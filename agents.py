import streamlit as st
from utils import load_data_agent , retrieve_data_agent ,save_uploaded_files
from llama_index.core.llms import ChatMessage, MessageRole

def main():
        
    with st.sidebar:
        selected_page = st.selectbox("Select Page", ["Upload Documents", "Chat with Agent"])         
        st.markdown("\n" * 30)
        
    if selected_page == "Upload Documents":
        
        st.header(" Load the Documents 📄")  
              
        with st.sidebar:
            uploaded_files = st.file_uploader("Upload documents", type=["docx", "pdf", "txt"], accept_multiple_files=True)
        
        agent_name = st.text_input("Enter agent name")

        if st.button("Create Agent"):
            if uploaded_files:
                file_paths = save_uploaded_files(uploaded_files)
                st.session_state["file_paths"] = file_paths 
                agent = load_data_agent()
                response = agent.chat(f"Load the documents {file_paths},and Create an Agent with the name {agent_name}")
                st.success(response.response,icon="✅")
            else:
                st.error("Please upload at least one document.")
    
    elif selected_page == "Chat with Agent":
        
        st.header("💬 Chat with your Documents 📄") 
        
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "agent",
                                      "content": "What questions do you have regarding the uploaded "
                                                 "Documents ? 📄"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.session_state.chat_history.append(ChatMessage(
                    role=MessageRole.USER,
                    content=prompt,
                ))
                st.write(prompt)
                
        if "file_paths" in st.session_state:
            file_paths = st.session_state["file_paths"]
            
            if st.session_state.messages[-1]["role"] != "agent":
                with st.chat_message("agent"):
                    with st.spinner("Thinking ... "):
                        retrieve_agent = retrieve_data_agent(files=file_paths)  
                        response = retrieve_agent.chat(prompt).response
                        st.write(response)
                message = {"role": "agent", "content": response}
                st.session_state.messages.append(message)

        else:
            st.warning("Please upload documents and create an agent first.")  
                
if __name__ == "__main__":
    main()
