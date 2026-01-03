from ollama import chat

_main_model = 'llama3.1:8b'

def get_temperature(city: str) -> str:
  """Get the current temperature for a city
  
  Args:
    city: The name of the city

  Returns:
    The current temperature for the city
  """
  temperatures = {
    "New York": "22°C",
    "London": "15°C",
    "Tokyo": "18°C"
  }
  return temperatures.get(city, "Unknown")

def get_conditions(city: str) -> str:
  """Get the current weather conditions for a city
  
  Args:
    city: The name of the city

  Returns:
    The current weather conditions for the city
  """
  conditions = {
    "New York": "Partly cloudy",
    "London": "Rainy",
    "Tokyo": "Sunny"
  }
  return conditions.get(city, "Unknown")


messages = [{'role': 'user', 'content': 'What are the current weather conditions and temperature in New York and London?'}]

# The python client automatically parses functions as a tool schema so we can pass them directly
# Schemas can be passed directly in the tools list as well 
response = chat(model=_main_model, messages=messages, tools=[get_temperature, get_conditions])

# add the assistant message to the messages
messages.append(response.message)
if response.message.tool_calls:
  # process each tool call 
  print(response.message.tool_calls)
  for call in response.message.tool_calls:

    # execute the appropriate tool
    if call.function.name == 'get_temperature':
      result = get_temperature(**call.function.arguments)
    elif call.function.name == 'get_conditions':
      result = get_conditions(**call.function.arguments)
    else:
      result = 'Unknown tool'
    # add the tool result to the messages
    messages.append({'role': 'tool',  'tool_name': call.function.name, 'content': str(result)})

  # generate the final response
  final_response = chat(model=_main_model, messages=messages, tools=[get_temperature, get_conditions])
  print(final_response.message.content)