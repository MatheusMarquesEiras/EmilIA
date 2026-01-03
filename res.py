from ollama import Client
from dataclasses import dataclass
from typing import Optional

_main_url = "http://localhost:11434"
_aux_url = "http://localhost:11435"
_main_model = 'llama3.1:8b'
_aux_model = 'granite3.2-vision'

# @dataclass
# class UserMessage:
#     message: str
#     tag: str
#     img: Optional[str] = None

#     def format(self):
#         data =  {
#             'role': 'user',
#             'content': f"{self.tag} {self.message}"
#         }
    
#         if self.img is not None:
#             data['image_url'] = self.img 
            
#         return data
    
# @dataclass
# class AIMessage:
#     message: str

#     def format(self):
#         return {
#             'role': 'assistant',
#             'content': self.message
#         }

# @dataclass
# class System:
#     message: str

#     def format(self):
#         return {
#             'role': 'system',
#             'content': self.message
#         }

# @dataclass
# class Tool:
#     name: str
#     result: str

#     def format(self):
#         return {
#             'role': 'tool',
#             'tool_name': self.name, 
#             'content': self.result
#         }

#     def is_empty(self):
#         return self.name == ''
    
#     def __repr__(self):
#         return f'{{tool={self.name[:15]}...}}'

from ollama import chat

@dataclass
class System:
    message: str

    def format(self):
        return {
            'role': 'system',
            'content': self.message
        }

@dataclass
class UserMessage:
    message: str
    tag: str
    img: Optional[str] = None

    def format(self):
        data =  {
            'role': 'user',
            'content': f"{self.tag} {self.message}"
        }
    
        if self.img is not None:
            data['image_url'] = self.img 
            
        return data

with open('sys.txt', 'r', encoding='utf-8') as file:
    sys = file.read()

sys = System(sys)
user_message = UserMessage(message='responda "sim"', tag='[Test]')

# print([sys.format(), user_message.format()])

stream = chat(
  model=_main_model,
  messages=[sys.format(), user_message.format()],
  stream=True,
)

content = ''
for chunk in stream:
  if chunk.message.content:
    print(chunk.message.content, end='', flush=True)

  