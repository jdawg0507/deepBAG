import google.generativeai as genai
from google.protobuf.struct_pb2 import Struct
from google.protobuf import json_format
import os

genai.configure(api_key=os.environ.get('GEMINI_API_KEY', 'AIzaSyDGvmWXOZRTyqWl_MhcYEAU2jvKV5mS8vA'))
model = genai.GenerativeModel('gemini-2.5-flash')
chat = model.start_chat()

# Simulate a function call in history
chat.history.append(genai.protos.Content(role='user', parts=[genai.protos.Part(text='test')]))
chat.history.append(genai.protos.Content(role='model', parts=[genai.protos.Part(function_call=genai.protos.FunctionCall(name='test', args={}))]))

# Test protobuf struct conversion
summary = {"rows": 0, "data": []}
s = Struct()
json_format.ParseDict({"result": summary}, s)

part = genai.protos.Part(
    function_response=genai.protos.FunctionResponse(
        name='test',
        response=s
    )
)

print("Sending part:", part)
try:
    response = chat.send_message(part)
    print("Success:", response.text)
except Exception as e:
    print("Error:", repr(e))
