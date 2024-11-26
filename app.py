import streamlit as st
from pathlib import Path
import base64

# Initial page config
st.set_page_config(
    page_title="Streamlit Cheat Sheet",
    layout="wide",
)

def main():
    cs_header()
    cs_tabs()

def img_to_bytes(img_path):
    try:
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except FileNotFoundError:
        return ''

# Header
def cs_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        img_data = img_to_bytes("logomark_website.png")  # Ensure this image is in the same directory
        if img_data:
            st.image(f"data:image/png;base64,{img_data}", width=50)
    with col2:
        st.title("📘 Streamlit Cheat Sheet")
        st.caption(
            """
            **Author**: [Daniel Lewis](https://daniellewisdl.github.io/)  
            **Contributors**: [Arnaud Miribel](https://github.com/arnaudmiribel), [Akrolsmir](https://github.com/akrolsmir), [Nathan Carter](https://github.com/nathancarter)  
            **Version**: 1.25.0 | **Date**: 20 August 2023
            """
        )
    st.markdown("---")

# Horizontal Tabs
def cs_tabs():
    tabs = st.tabs(["💬 Basics", "🖼️ Display", "🎛️ Widgets", "⚡ Performance", "📡 Data Sources", "🚀 Status & Progress"])

    with tabs[0]:
        display_basics()

    with tabs[1]:
        display_media()

    with tabs[2]:
        interactive_widgets()

    with tabs[3]:
        optimize_performance()

    with tabs[4]:
        connect_data_sources()

    with tabs[5]:
        display_status_progress()

# Basics Tab
def display_basics():
    st.header("💬 Basics")

    with st.expander("🟦 Importing Streamlit"):
        st.code("""
# Install Streamlit
$ pip install streamlit

# Import convention
>>> import streamlit as st
        """)

    with st.expander("🟩 Command Line Commands"):
        st.code("""
$ streamlit --help
$ streamlit run your_script.py
$ streamlit hello
$ streamlit cache clear
$ streamlit --version
        """)

    with st.expander("🟨 Layout and Configuration"):
        st.code("""
st.set_page_config(
    page_title="My App",
    layout="wide",
    initial_sidebar_state="expanded"
)
        """)

    with st.expander("🟧 Magic Commands"):
        st.code("""
# Magic commands automatically print variables and text
'_This_ is **Markdown**'
x = 42
'x:', x
        """)

    with st.expander("🟥 Control Flow"):
        st.code("""
# Stop execution
st.stop()

# Rerun script
st.experimental_rerun()
        """)

# Display Tab
def display_media():
    st.header("🖼️ Display Data and Media")

    with st.expander("🟩 Display Data"):
        st.code("""
st.dataframe(my_dataframe)
st.table(my_dataframe.iloc[:10])
st.json({'key': 'value'})
st.metric(label="Temperature", value="273 K", delta="1.2 K")
        """)

    with st.expander("🟨 Display Media"):
        st.code("""
st.image("path_to_image.png")
st.audio("path_to_audio.mp3")
st.video("path_to_video.mp4")
        """)

    with st.expander("🟧 Text Elements"):
        st.code("""
st.title("My App Title")
st.header("Section Header")
st.subheader("Subsection Header")
st.text("Simple text display")
st.markdown("This is **Markdown**")
st.latex(r'''e^{i\pi} + 1 = 0''')
        """)

# Widgets Tab
def interactive_widgets():
    st.header("🎛️ Interactive Widgets")

    with st.expander("🟩 Basic Widgets"):
        st.code("""
st.button("Click me!")
st.checkbox("Check me out")
st.radio("Pick one", ["Option 1", "Option 2"])
st.selectbox("Select one", ["A", "B", "C"])
st.multiselect("Select multiple", ["X", "Y", "Z"])
st.slider("Slide me", min_value=0, max_value=100)
        """)

    with st.expander("🟨 Input Widgets"):
        st.code("""
st.text_input("Enter text")
st.number_input("Enter a number")
st.text_area("Enter multi-line text")
st.date_input("Pick a date")
st.time_input("Pick a time")
        """)

    with st.expander("🟧 File and Camera Input"):
        st.code("""
st.file_uploader("Upload a file")
st.camera_input("Capture an image")
        """)

# Performance Tab
def optimize_performance():
    st.header("⚡ Optimize Performance")

    with st.expander("🟩 Caching Data"):
        st.code("""
@st.cache_data
def expensive_computation():
    # Perform expensive computations
    return result

result = expensive_computation()
        """)

    with st.expander("🟨 Caching Resources"):
        st.code("""
@st.cache_resource
def expensive_resource():
    # Create and return non-data objects
    return resource

resource = expensive_resource()
        """)

    with st.expander("🟧 Deprecated Caching"):
        st.code("""
@st.cache
def old_caching_method():
    # Perform computations
    return data
        """)

# Data Sources Tab
def connect_data_sources():
    st.header("📡 Connect to Data Sources")

    with st.expander("🟩 Database Connections"):
        st.code("""
from sqlalchemy import create_engine

# Create connection
engine = create_engine("sqlite:///example.db")
query_result = pd.read_sql("SELECT * FROM my_table", con=engine)
        """)

    with st.expander("🟨 Snowflake Integration"):
        st.code("""
# Snowflake connector example
import snowflake.connector

conn = snowflake.connector.connect(
    user="your_username",
    password="your_password",
    account="your_account"
)
        """)

# Status and Progress Tab
def display_status_progress():
    st.header("🚀 Status and Progress")

    with st.expander("🟩 Progress Bars"):
        st.code("""
progress = st.progress(0)

for i in range(100):
    time.sleep(0.1)
    progress.progress(i + 1)
        """)

    with st.expander("🟨 Notifications"):
        st.code("""
st.balloons()
st.snow()
st.toast("Hello, World!")
        """)

    with st.expander("🟧 Error Messages"):
        st.code("""
st.error("An error occurred!")
st.warning("This is a warning!")
st.info("For your information...")
st.success("Operation was successful!")
        """)

# Run Main
if __name__ == "__main__":
    main()
