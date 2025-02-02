

import os
import gradio as gr
import ai_gradio

from gradio_webrtc import WebRTC

# Use WebRTC with mode="video"



WebRTC(mode="video", label="Live Video")

# Create a vision-enabled interface with camera support
# gr.load(
#     name='gemini:gemini-2.0-flash-exp',
#     src=ai_gradio.registry,
#     camera=True,
# ).launch()

# gr.load(
#     name='gemini:gemini-pro', # 'gemini:gemini-2.0-flash-exp',  # or other supported models
#     src=ai_gradio.registry,
#     title='gemini Agent',
#     description='AI agent powered by Langchain'
# ).launch()


# gr.load(
#     name='browser:o3-mini-2025-01-31',
#     src=ai_gradio.registry,
#     title='Local Agent',
# ).launch()

# Create a coding assistant with Gemini
# gr.load(
#     name='gemini:gemini-2.0-flash-thinking-exp-1219',  # or 'openai:gpt-4-turbo', 'anthropic:claude-3-opus'
#     src=ai_gradio.registry,
#     coder=True,
#     title='Gemini Code Generator',
# ).launch()

# with gr.Blocks() as demo:
#     with gr.Tab("Text"):
#         gr.load('openai:gpt-4', src=ai_gradio.registry)
#     with gr.Tab("Vision"):
#         gr.load('gemini:gemini-pro-vision', src=ai_gradio.registry)
#     with gr.Tab("Code"):
#         gr.load('gemini:gemini-pro', src=ai_gradio.registry)

# demo.launch()

# gr.load('gemini:gemini-pro-vision', 
#         src=ai_gradio.registry,

#         ).launch()

# gr.load(
#     name='gemini:gemini-2.0-flash-exp',
#     src=ai_gradio.registry,
#     enable_voice=True,
#     title='AI Voice Assistant'
# ).launch()



# Create a chat interface with Swarms
gr.load(
    name='swarms:gpt-4',  # or other OpenAI models
    src=ai_gradio.registry,
    agent_name="Stock-Analysis-Agent",  # customize agent name
    title='Swarms Chat',
    description='Chat with an AI agent powered by Swarms'
).launch()