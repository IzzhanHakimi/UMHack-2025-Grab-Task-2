import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime
import time
import openai
import ast
import pandas as pd
import base64
import re
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import time as t

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Image

from explore_data import revenue_per_date, most_ordered_product, total_revenue, avg_wait_and_ready_time, order_per_date, merged_df, total_orders
from visualisation import sales_trend, best_product, orders_trend
from urllib.parse import unquote
from real_time import start_streaming_thread, start_inv_sim, seq, shared_state
from streamlit_autorefresh import st_autorefresh

def generate_chat_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    Story = []
    
    # Add title
    title_style = styles["Title"]
    title = Paragraph("Chat History Report", title_style)
    Story.append(title)
    Story.append(Spacer(1, 24))
    
    qna_count = 0  # Track Q&A pairs per page
    img_counter = 0  # Track graph images
    
    chat_history = st.session_state.chat_history
    
    # Skip initial bot message if exists
    start_index = 0
    if chat_history and chat_history[0][0] == "Bot":
        start_index = 1
    
    for i in range(start_index, len(chat_history), 2):
        if i+1 >= len(chat_history):
            break
        
        if qna_count == 2:
            # Add page break
            Story.append(Spacer(1, 12))
            Story.append(PageBreak())
            qna_count = 0
        
        user_entry = chat_history[i]
        bot_entry = chat_history[i+1]
        
        # Question
        question_text = f"<b>Question:</b> {user_entry[1]}"
        p_question = Paragraph(question_text, styles["BodyText"])
        Story.append(p_question)
        Story.append(Spacer(1, 12))
        
        # Answer
        answer_text = f"<b>Answer:</b> {bot_entry[1]}"
        p_answer = Paragraph(answer_text, styles["BodyText"])
        Story.append(p_answer)
        Story.append(Spacer(1, 24))
        
        # Handle graph
        if bot_entry[2] is not None:
            try:
                img_path = f"graph{img_counter}.png"
                img = Image(img_path, width=5*inch, height=3*inch)
                Story.append(img)
                Story.append(Spacer(1, 24))
                img_counter += 1
                
                # If image takes too much space, force new page
                if qna_count == 1:
                    Story.append(PageBreak())
                    qna_count = 0
            except Exception as e:
                error_text = Paragraph("<i>[Graph unavailable]</i>", styles["Italic"])
                Story.append(error_text)
        
        qna_count += 1
    
    doc.build(Story)
    buffer.seek(0)
    return buffer.getvalue()


FORBIDDEN_NAMES = { "sys", "subprocess", "shutil", "__import__"}

def is_code_safe(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print("Syntax error in code:", e)
        return False

    for node in ast.walk(tree):
        # Check for dangerous imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if any(name.split('.')[0] in FORBIDDEN_NAMES for name in names):
                print(f"Found forbidden import in code: {names}")
                return False

        # Check for dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}:
                print(f"Found dangerous function call: {node.func.id}")
                return False

    return True

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to get AI response using OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_ai_reply(query):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

start_streaming_thread('Bagel Bros')
start_streaming_thread('Noodle Nest')

if "inv_status" not in st.session_state:
    st.session_state.inv_status = None

start_inv_sim(seq)

new_status = shared_state.get("inv_status")
if new_status is not None:
    st.session_state.inv_status = new_status

# Main dashboard code
st.set_page_config(page_title="Seller Dashboard", layout="wide")

if st.button('Click to Refresh'):
    st.rerun()

#refresh_count = st_autorefresh(interval=12500, limit=None, key="inventory_autorefresh")

# Sidebar layout for the chatbot
shop_param = st.query_params.get_all('shop')
if shop_param:
    shop = unquote(shop_param[0])
else:
    shop = 'Bagel Bros'

time_param = st.query_params.get_all('time')
if shop_param:
    time = unquote(time_param[0])
else:
    time = 'Today'

if "shop" not in st.session_state:
    st.session_state.shop = 'Bagel Bros'
else:
    shop = st.session_state.shop

if "time" not in st.session_state:
    st.session_state.time = 'Today'
else:
    time = st.session_state.time

df_merged_live = merged_df(shop, None, live=True)
df_past_data = merged_df(shop, None, all=True)
df_merged = pd.concat([df_past_data, df_merged_live], ignore_index=True)

additional_context = df_merged["item_name"].unique()

datetime_columns = ['order_time', 'driver_arrival_time', 'driver_pickup_time', 'delivery_time']

for col in datetime_columns:
    df_merged[col] = pd.to_datetime(df_merged[col],errors='coerce')

def chat_bot_output(user_input):

    #user_input = "Based on the data for the past 14 days (today 17th June 2023), which hour usually causes operational bottlenecks and how to avoid that efficiently"
    prompt = f'''

    You are a RAG LLM Agent that will answer questions about a csv file, the csv file is already readed and right now the csv file is in a variable called df_merged . This dataset is data about food delivery services
    The dataset contains columns which is :

    - "order_id" (this is the id of each order, this is not a primary key because here some rows have the same order_id but different item_id which means that the order consist of more than one meal)
    - "order_time" (this is a datetime column that shows when the order is placed by the customer)
    - "driver_arrival_time" (this is a datetime column that shows when the delivery guy arrives at a restaurant)
    - "driver_pickup_time" (this is a datetime column that shows when the delivery guy able to pick up the food from the restaurant to send to the customer)
    - "delivery_time" (this is a datetime column that shows when the delivery guy arrives at the customer's house to deliver the food)
    - "eater_id" (this is the id of each customer)
    - "order_ready" (this just the period (in minutes) between the column driver_pickup_time and driver_order_time)
    - "driver_arrival_period" (this is the period (in minutes) between the column order_time and driver_arrival_time)
    - "driver_waiting_time" (this is the period (in minutes) between the column driver_arrival_time and driver_pickup_time)
    - "item_id" (this is the id of each meal/items)
    - merchant_id_x (this is the id of the merchant/restaurant)
    - cuisine_tag  (this is the category of each meal, like "western","breakfast",etc..)
    - item_name (this is the name of the meal/item)
    - item_price (this is the price of the meal/item)
    - ingredients (this is an array of ingredients used to make the meal but this is a string column, example values are like "["Onions","Rice","Egg"]" so you need to handle that)

    additional information, These are all the products that they sold (unique values in the item_name column ):{additional_context}

    Based on this, write me a python code to answer this question {user_input} , at the end of the code store the final output in a variable called Final_Output (If you think that the question cannot be answered by the data, just say "Sorry I dont know") ,
    This  Final_Output variable will be send to an api to OpenAI to do some analysis so it cannot be None so make sure th structure of this variable is suitable for analysis
    This Final_Output also CANNOT BE A SENTENCE because we cant perform data analysis using a sentence,
    display your final output using print statements, or display a matplotlib graph (if the user wants a graph)
    If you displayed a graph, put it in a variable called Final_Graph (in plt format), if you are not displaying a graph, just set Final_Graph = None
    If the user asks to do predictions and ML models are involved, include a gridsearch cv code to ensure 
    Import necessary Libraries and Your Output should only be PYTHON CODE and nothing else
    For your information this dataset only contains data in the year 2023, and today is 17th June 2023
    '''

    extra_string =f'''

import openai
import os
import matplotlib.pyplot as plt

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Case 1: If Final_Output is a matplotlib figure (assumed to be shown already)
if isinstance(Final_Output, plt.Figure):
    print("Case 1")
    image_path = "plot_output.png"
    Final_Output.savefig(image_path)
    final_output_str = "A graph has been generated and saved as 'plot_output.png'."
    import base64
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = "Based on this image, analyse it and give a proper answer to the question '{user_input}' (answer the question using the language that is used in the query)"

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    # Encode the image
    base64_image = encode_image(image_path)

    # Create chat completion
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=300,
            messages=[
                {{
                    "role": "user",
                    "content": [
                        {{
                            "type": "text",
                            "text": prompt
                        }},
                        {{
                            "type": "image_url",
                            "image_url": {{
                                "url": f"data:image/jpeg;base64,{{base64_image}}"
                            }}
                        }}
                    ]
                }}
            ]
        )
        print("----------------------------------")
        Final_Reply = chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occured: {{e}}")
    
# Case 2: If Final_Output is a plot shown via plt (stateful)


# Case 3: If it's just textual or tabular data
else:
    print("Case 3")
    final_output_str = str(Final_Output)
    query = f"""Based on final output : {{Final_Output}}, provide a proper answer to the query which is '{user_input}' (answer the question using the language that is used in the query)"""
    print(query)
    print("----------------------------------")
    Final_Reply = get_ai_reply(query)
'''

    Final_Output = "Bellow"
    Final_Graph = None


    #user_input = "Produk apa paling laku do"
    code = get_ai_reply(prompt)
    code = code + extra_string

    code = re.sub(r'```python', '', code)
    code = re.sub(r'```', '', code)
    with open("logs.txt", "w") as f:
            f.write(code)


    # Remove code block markers
    code = re.sub(r'```python', '', code)
    code = re.sub(r'```', '', code)

    with open("logs.txt", "w") as f:
        f.write(code)


    exec_globals={}
    try:
        if is_code_safe(code):
            exec_globals["df_merged"] = df_merged
            exec_globals["get_ai_reply"] = get_ai_reply
            exec(code,exec_globals)
            Final_Output = exec_globals.get('Final_Reply')
            Final_Graph = exec_globals.get('Final_Graph')
        else:
            print("There is something wrong")
    except Exception as e:
        print(f"Error: {e}")


    print (Final_Output,Final_Graph)

    return Final_Output,Final_Graph

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bot_typing" not in st.session_state:
    st.session_state.bot_typing = False
if "graph_i" not in st.session_state:
    st.session_state.graph_i = 0

# Ensure bot gives an opening message
if not st.session_state.chat_history:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(("Bot", "Hello! How can I assist you today?",None, timestamp))

# Simulate handling user messages
def handle_user_message():
    user_input = st.session_state.chat_input.strip()
    
    if user_input:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append(("You", user_input,None, timestamp))
        st.session_state.chat_input = ""
        st.session_state.bot_typing = True
        response,graph = chat_bot_output(user_input)
        st.session_state.chat_history.append(("Bot", response,graph, datetime.now().strftime("%H:%M:%S")))
        if graph is not None :
            graph.savefig(f"graph{st.session_state.graph_i}.png", dpi=300, bbox_inches="tight")
            st.session_state.graph_i=st.session_state.graph_i+1
        st.session_state.bot_typing = False
        #st.rerun()

#print(st.session_state.chat_history)

# Custom CSS for the app layout
st.markdown("""
    <style>
        body, .stApp {
            background-color: #fffff;
            color: black;
        }
        [data-testid="stSidebar"] {
            background-color: #08543c;
            color: white;
            border-radius: 15px;
        }
        [data-testid="stTextInput"] div div input {
                    background-color: white;
                    border: 2px solid white;
                    color: black;
                }
            
        /* Main button styling */
        div[data-testid="stButton"] > button {
            background-color: #10543c !important;
            border: 2px solid white !important;
            color: white !important;
            transition: all 0.3s ease !important;
        }

        /* Hover effects */
        div[data-testid="stButton"] > button:hover {
            background-color: #0d4530 !important;
            border: 2px solid white !important;
            color: white !important;
            filter: brightness(90%);
        }

        /* Active/click state */
        div[data-testid="stButton"] > button:active {
            background-color: #0a3726 !important;
        }

        /* Focus state */
        div[data-testid="stButton"] > button:focus {
            box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.5) !important;
        }

        /* Disabled state */
        div[data-testid="stButton"] > button:disabled {
            background-color: #10543c88 !important;
            border-color: #ffffff88 !important;
            }
        [data-testid="stTextInput"] div div input {
                    background-color: white;
                    border: 2px solid white;
                    color: black;
        }
        .chat-box {
            max-height: 250px;
            overflow-y: auto;
            padding: 10px;
            background-color: #08543c;  /* new background color */
            border-radius: 10px;
            margin-bottom: 10px;
            color: white;  /* ensures text inside is white */
            border: 2px solid white;
        }
        .message {
            padding: 8px 12px;
            margin: 6px 0;
            border-radius: 10px;
            max-width: 90%;
            word-wrap: break-word;
            
        }
        .user {
            background-color: #a0c454;
            color: white;
            text-align: right;
            margin-left: auto;
            
            
        }
        .bot {
            background-color: #109c34;
            color: white;
            text-align: left;
            margin-right: auto;
            
            
        }
        .timestamp {
            font-size: 10px;
            color: gray;
        }
        
    </style>
""", unsafe_allow_html=True)

def fig_to_base64_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

with st.sidebar:
    if 'developer_mode' not in st.session_state:
        st.session_state.developer_mode = False
    
    if st.button('Developer Mode'):
        st.session_state.developer_mode = not st.session_state.developer_mode  # toggle on/off


    # Initialize session state if not present

    # Get query parameters from URL
    shops = ["Bagel Bros", "Noodle Nest"]

    # Retrieve the shop query parameter from the URL (if it exists)
    if "shop" not in st.query_params:
        # Default to "Bagel Bros" if no query parameter exists
        st.query_params.shop = "Bagel Bros"

    # Set the initial shop value from query params or session state
    #shop = st.query_params.shop


    # Button for Developer Mode
    if st.session_state.developer_mode:
        # When the button is pressed, show the dropdown
        # Show the dropdown

        st.session_state.setdefault("options", shops)


        def update_options():
            # move the selected option to the front of the list if it is not already
            if st.session_state.selected_option != st.session_state.options[0]:
                st.session_state.options.remove(st.session_state.selected_option)
                st.session_state.options.insert(0, st.session_state.selected_option)

        shop = st.selectbox(
            label='Select an option',
            options=st.session_state.options,
            key="selected_option",
            on_change=update_options,
        )

    # Update the shop value in query params
        st.session_state.shop = shop
        st.query_params.shop = shop


        # Dropdown 2
    #time = st.selectbox('Select from Dropdown 2:', ['Today', 'This Week', 'This Month'])


    # Initialize session state if not present

    # Get query parameters from URL
    times = ["Today", "This Week", "This Month"]

    # Retrieve the shop query parameter from the URL (if it exists)
    if "time" not in st.query_params:
        # Default to "Bagel Bros" if no query parameter exists
        st.query_params.time = "Today"

    # Set the initial shop value from query params or session state
    #shop = st.query_params.shop


    # Button for Developer Mode
    if st.session_state.developer_mode:
        # When the button is pressed, show the dropdown
        # Show the dropdown

        st.session_state.setdefault("times", times)


        def update_options():
            # move the selected option to the front of the list if it is not already
            if st.session_state.selected_time != st.session_state.times[0]:
                st.session_state.times.remove(st.session_state.selected_time)
                st.session_state.times.insert(0, st.session_state.selected_time)

        time = st.selectbox(
            label='Select an option',
            options=st.session_state.times,
            key="selected_time",
            on_change=update_options,
        )

    # Update the shop value in query params
        st.session_state.time = time
        st.query_params.time = time


    st.markdown("""
    <h1 style='text-align: center; font-size: 32px;'>🤖 Mex Assistant</h1>
""", unsafe_allow_html=True)
    st.markdown(
    "<hr style='border: 1px solid white;'>",
    unsafe_allow_html=True
    )
    # Scrollable chat history box
    with st.container():
        i=0
        #st.markdown('<div class="chat-box" style="max-height: 250px; overflow-y: auto;">', unsafe_allow_html=True)
        for sender, message, graph, timestamp in st.session_state.chat_history:
            sender_class = "user" if sender == "You" else "bot"
            align = "right" if sender == "You" else "left"
            if graph is not None:
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.image(f"graph{i}.png",use_container_width=True)
                    i=i+1
                with col2:
                    st.write("")

            st.markdown(
                f"""
                <div class="message {sender_class}">
                    <div>{message}</div>
                    <div class="timestamp" style="text-align: {align};">{timestamp}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Text input for user query
    st.markdown(
    "<hr style='border: 1px solid white;'>",
    unsafe_allow_html=True
    )

    st.text_input("",placeholder="Ask the chatbot here", key="chat_input", on_change=handle_user_message)
    #st.button("Personalized", use_container_width=True)
    #st.button("Download", use_container_width=True)

    st.markdown(
            """
            <style>
            /* Targeting the download button inside its container */
            div.stDownloadButton button {
                background-color: #10543c;
                color: white;
                border: 1px solid white;
                padding: 0.75rem 1.5rem;  /* Adjust padding as needed */
                font-size: 1rem;
            }
            div.stDownloadButton button:hover {
                background-color: #0e4532;  /* Slightly darker on hover */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    if st.button("Download Chat", use_container_width=True):
        pdf_bytes = generate_chat_pdf()
        st.download_button(
            label="Confirm Download",
            data=pdf_bytes,
            file_name="chat_history.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# Reusable card layout
def bordered_card(content, bg_color="#ffffff", text_color="black"):
    st.markdown(
        f"""
        <div style="
            border: 2px solid #28a745;
            border-radius: 15px;
            padding: 20px;
            background-color: {bg_color};
            color: {text_color};
            height: 100%;
        ">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )

# Top bar
top_left, spacer, profile, settings, logout = st.columns([3, 5, 1, 1, 1])
with top_left:
    st.image("grab.png", width=150)
with profile:
    st.write("**Profile**")
with settings:
    st.write("**Settings**")
with logout:
    st.write(":red[**Logout**]")

st.markdown("---")

# Dashboard Title
st.markdown('<h1 style="color:black;">Seller Dashboard</h1>', unsafe_allow_html=True)


# Top row
driver_wait, order_prep = avg_wait_and_ready_time(shop)

col1, col6, col7, col2 = st.columns(4)
with col1:
    revenue = total_revenue(shop, live=True, time=time)
    revenue = "{:,.2f}".format(revenue)

    bordered_card(f"""<p><strong>{time}'s Total Sale</strong></p>
                  <h3 style='color:black;'>RM{revenue}</h3>""")

with col6:
    orders = total_orders(shop, time=time, live=True)
    bordered_card(f"""<p><strong>{time}'s Total Orders</strong></p>
                  <h3 style='color:black;'>{orders}</h3>""")

with col7:
    order_prep = "{:,.2f}".format(order_prep)
    bordered_card(f"""<p><strong>Time to Prepare Order</strong></p>
                  <h3 style='color:black;'>{order_prep} minutes</h3>""")

with col2:
    #bordered_card-bordered_card_with_plot("<strong>Sales Trend</strong><br>", )
    driver_wait = "{:,.2f}".format(driver_wait)
    bordered_card(f"""<p><strong>Driver Waiting Time</strong></p>
                  <h3 style='color:black;'>{driver_wait} minutes</h3>""")
    
    #st.line_chart([100, 150, 200, 180, 220, 250])

#print(df_merged)

if "noti" not in st.session_state:
    st.session_state.noti = []

if st.session_state.inv_status == 0:
    st.toast("🔄 Inventory is running out! Do restock ASAP")
    st.session_state.noti.append("🔄 Inventory is running out! Do restock ASAP")

st.write(" ")
st.write(" ")
st.write(" ")

col3, col4 = st.columns([3, 1])
with col3:
    bordered_card("<strong>Orders Trend</strong><br>")
    order_data = order_per_date(shop, time)
    plt = orders_trend(order_data, time)
    st.pyplot(plt)
with col4:
    bordered_card("<strong>Product Performance</strong><br>")
    quantity_data = most_ordered_product(shop, time)
    plt = best_product(quantity_data, time)
    st.pyplot(plt)

# Middle row
col8, col5 = st.columns([3, 1])
with col8:
    bordered_card("<strong>Revenue Trend</strong><br>")
    tr_data = revenue_per_date(shop, time)
    plt = sales_trend(tr_data, time)
    st.pyplot(plt)

with col5:
    noti_text = "No new notifications."

    if len(st.session_state.noti) > 0:
        noti_text = ""
        for i in range(len(st.session_state.noti)):
            noti_text = "<br>".join(st.session_state.noti)
            
    bordered_card(f"<strong>Notification Centre</strong><br><p>{noti_text}</p>", bg_color="#28a745", text_color="white")