"""
Streamlit Cheat Sheet

App to summarize Streamlit docs v1.25.0

There is also an accompanying png and pdf version

https://github.com/daniellewisDL/streamlit-cheat-sheet

v1.25.0
20 August 2023

Author:
    @daniellewisDL : https://github.com/daniellewisDL

Contributors:
    @arnaudmiribel : https://github.com/arnaudmiribel
    @akrolsmir : https://github.com/akrolsmir
    @nathancarter : https://github.com/nathancarter

"""

import streamlit as st
from pathlib import Path
import base64
import pandas

# Initial page config
st.set_page_config(
     page_title='Streamlit Cheat Sheet',
     layout="wide",
     initial_sidebar_state="expanded",
)

def main():
    cs_header()
    cs_body()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def cs_header():
    # Custom CSS for styling
    st.markdown('''
        <style>
            .header {
                display: flex;
                align-items: center;
                background-color: #FF4B4B;
                padding: 10px;
            }
            .header img {
                width: 50px;
                margin-right: 10px;
            }
            .header h1 {
                color: #FFFFFF;
            }
            .block-container {
                padding-top: 0rem;
            }
        </style>
    ''', unsafe_allow_html=True)

    logo = img_to_bytes("logomark_website.png")
    st.markdown(f'''
        <div class="header">
            <img src="data:image/png;base64,{logo}">
            <h1>Streamlit Cheat Sheet</h1>
        </div>
    ''', unsafe_allow_html=True)

##########################
# Main body of cheat sheet
##########################

def cs_body():

    # Create horizontal tabs for navigation
    tabs = st.tabs(["Display text", "Display data", "Display media", "Columns", "Tabs", "Control flow",
                    "Display interactive widgets", "Mutate data", "Display code",
                    "Placeholders, help, and options", "Connect to data sources",
                    "Optimize performance", "Display progress and status"])

    with tabs[0]:
        display_text()
    with tabs[1]:
        display_data()
    with tabs[2]:
        display_media()
    with tabs[3]:
        display_columns()
    with tabs[4]:
        display_tabs_section()
    with tabs[5]:
        display_control_flow()
    with tabs[6]:
        display_widgets()
    with tabs[7]:
        display_mutate_data()
    with tabs[8]:
        display_display_code()
    with tabs[9]:
        display_placeholders()
    with tabs[10]:
        display_data_sources()
    with tabs[11]:
        display_performance()
    with tabs[12]:
        display_progress()

# Define each section as a function

def display_text():
    st.subheader('Display text')
    st.code('''
st.text('Fixed width text')
st.markdown('_Markdown_') # see #*
st.caption('Balloons. Hundreds of them...')
st.latex(r\'\'\' e^{i\pi} + 1 = 0 \'\'\')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')

# * optional kwarg unsafe_allow_html = True
    ''')
    st.markdown('<small>Learn more about [displaying text](https://docs.streamlit.io/library/api-reference/write-magic)</small>', unsafe_allow_html=True)

def display_data():
    st.subheader('Display data')
    st.code('''
st.dataframe(my_dataframe)
st.table(data.iloc[0:10])
st.json({'foo':'bar','fu':'ba'})
st.metric(label="Temp", value="273 K", delta="1.2 K")
    ''')
    st.markdown('<small>Learn more about [displaying data](https://docs.streamlit.io/library/api-reference/data)</small>', unsafe_allow_html=True)

def display_media():
    st.subheader('Display media')
    st.code('''
st.image('./header.png')
st.audio(data)
st.video(data)
    ''')
    st.markdown('<small>Learn more about [displaying media](https://docs.streamlit.io/library/api-reference/media)</small>', unsafe_allow_html=True)

def display_columns():
    st.subheader('Columns')
    st.code('''
col1, col2 = st.columns(2)
col1.write('Column 1')
col2.write('Column 2')

# Three columns with different widths
col1, col2, col3 = st.columns([3,1,1])
# col1 is wider

# Using 'with' notation:
>>> with col1:
>>>     st.write('This is column 1')
    ''')
    st.markdown('<small>Learn more about [columns](https://docs.streamlit.io/library/api-reference/layout/st.columns)</small>', unsafe_allow_html=True)

def display_tabs_section():
    st.subheader('Tabs')
    st.code('''
# Insert containers separated into tabs:
>>> tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
>>> tab1.write("this is tab 1")
>>> tab2.write("this is tab 2")

# You can also use "with" notation:
>>> with tab1:
>>>   st.radio('Select one:', [1, 2])
    ''')
    st.markdown('<small>Learn more about [tabs](https://docs.streamlit.io/library/api-reference/layout/st.tabs)</small>', unsafe_allow_html=True)

def display_control_flow():
    st.subheader('Control flow')
    st.code('''
# Stop execution immediately:
st.stop()
# Rerun script immediately:
st.experimental_rerun()

# Group multiple widgets:
>>> with st.form(key='my_form'):
>>>   username = st.text_input('Username')
>>>   password = st.text_input('Password')
>>>   st.form_submit_button('Login')
    ''')
    st.markdown('<small>Learn more about [control flow](https://docs.streamlit.io/library/api-reference/control-flow)</small>', unsafe_allow_html=True)

def display_widgets():
    st.subheader('Display interactive widgets')
    st.code('''
st.button('Hit me')
st.data_editor('Edit data', data)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')
st.download_button('On the dl', data)
st.camera_input("一二三,茄子!")
st.color_picker('Pick a color')
    ''')
    st.code('''
# Use widgets' returned values in variables
>>> for i in range(int(st.number_input('Num:'))): foo()
>>> if st.sidebar.selectbox('I:',['f']) == 'f': b()
>>> my_slider_val = st.slider('Quinn Mallory', 1, 88)
>>> st.write(my_slider_val)
    ''')
    st.code('''
# Disable widgets to remove interactivity:
>>> st.slider('Pick a number', 0, 100, disabled=True)
    ''')
    st.markdown('<small>Learn more about [interactive widgets](https://docs.streamlit.io/library/api-reference/widgets)</small>', unsafe_allow_html=True)

def display_mutate_data():
    st.subheader('Mutate data')
    st.code('''
# Add rows to a dataframe after
# showing it.
>>> element = st.dataframe(df1)
>>> element.add_rows(df2)

# Add rows to a chart after
# showing it.
>>> element = st.line_chart(df1)
>>> element.add_rows(df2)
    ''')
    st.markdown('<small>Learn more about [mutating data](https://docs.streamlit.io/library/api-reference/data/st.dataframe)</small>', unsafe_allow_html=True)

def display_display_code():
    st.subheader('Display code')
    st.code('''
st.echo()
>>> with st.echo():
>>>     st.write('Code will be executed and printed')
    ''')
    st.markdown('<small>Learn more about [displaying code](https://docs.streamlit.io/library/api-reference/code)</small>', unsafe_allow_html=True)

def display_placeholders():
    st.subheader('Placeholders, help, and options')
    st.code('''
# Replace any single element.
>>> element = st.empty()
>>> element.line_chart(...)
>>> element.text_input(...)  # Replaces previous.

# Insert out of order.
>>> elements = st.container()
>>> elements.line_chart(...)
>>> st.write("Hello")
>>> elements.text_input(...)  # Appears above "Hello".

st.help(pandas.DataFrame)
st.get_option(key)
st.set_option(key, value)
st.set_page_config(layout='wide')
st.experimental_show(objects)
st.experimental_get_query_params()
st.experimental_set_query_params(**params)
    ''')
    st.markdown('<small>Learn more about [placeholders and options](https://docs.streamlit.io/library/api-reference/layout)</small>', unsafe_allow_html=True)

def display_data_sources():
    st.subheader('Connect to data sources')
    st.code('''
st.experimental_connection('pets_db', type='sql')
conn = st.experimental_connection('sql')
conn = st.experimental_connection('snowpark')

>>> class MyConnection(ExperimentalBaseConnection[myconn.MyConnection]):
>>>    def _connect(self, **kwargs) -> MyConnection:
>>>        return myconn.connect(**self._secrets, **kwargs)
>>>    def query(self, query):
>>>       return self._instance.query(query)
    ''')
    st.markdown('<small>Learn more about [connecting to data sources](https://docs.streamlit.io/library/api-reference/connections)</small>', unsafe_allow_html=True)

def display_performance():
    st.subheader('Optimize performance')
    st.write('Cache data objects')
    st.code('''
# E.g. Dataframe computation, storing downloaded data, etc.
>>> @st.cache_data
... def foo(bar):
...   # Do something expensive and return data
...   return data
# Executes foo
>>> d1 = foo(ref1)
# Does not execute foo
# Returns cached item by value, d1 == d2
>>> d2 = foo(ref1)
# Different arg, so function foo executes
>>> d3 = foo(ref2)
# Clear all cached entries for this function
>>> foo.clear()
# Clear values from *all* in-memory or on-disk cached functions
>>> st.cache_data.clear()
    ''')
    st.write('Cache global resources')
    st.code('''
# E.g. TensorFlow session, database connection, etc.
>>> @st.cache_resource
... def foo(bar):
...   # Create and return a non-data object
...   return session
# Executes foo
>>> s1 = foo(ref1)
# Does not execute foo
# Returns cached item by reference, s1 == s2
>>> s2 = foo(ref1)
# Different arg, so function foo executes
>>> s3 = foo(ref2)
# Clear all cached entries for this function
>>> foo.clear()
# Clear all global resources from cache
>>> st.cache_resource.clear()
    ''')
    st.write('Deprecated caching')
    st.code('''
>>> @st.cache
... def foo(bar):
...   # Do something expensive in here...
...   return data
>>> # Executes foo
>>> d1 = foo(ref1)
>>> # Does not execute foo
>>> # Returns cached item by reference, d1 == d2
>>> d2 = foo(ref1)
>>> # Different arg, so function foo executes
>>> d3 = foo(ref2)
    ''')
    st.markdown('<small>Learn more about [optimizing performance](https://docs.streamlit.io/library/advanced-features/caching)</small>', unsafe_allow_html=True)

def display_progress():
    st.subheader('Display progress and status')
    st.code('''
# Show a spinner during a process
>>> with st.spinner(text='In progress'):
>>>   time.sleep(3)
>>>   st.success('Done')

# Show and update progress bar
>>> bar = st.progress(50)
>>> time.sleep(3)
>>> bar.progress(100)

st.balloons()
st.snow()
st.toast('Mr Stay-Puft')
st.error('Error message')
st.warning('Warning message')
st.info('Info message')
st.success('Success message')
st.exception(e)
    ''')
    st.markdown('<small>Learn more about [progress and status](https://docs.streamlit.io/library/api-reference/status)</small>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
