# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:29:58 2023

@author: alonh

This program let you extract, analyze and display data from the World-Bank API, 
using GPT and openAI function-calling.
"""

import openai
import json
import inspect
from matplotlib import pyplot as plt
import wbgapi as wb

openai.api_key = "Enter your OpenAI-Key Here"

# Send requests to the World-Bank, using wbgapi API:
def get_world_bank_data(country: str, field: str, years: list[int]):
    '''
    Get data from the World-Bank website about a given country's economical 
    field (like GDP, growth rates etc) in given years.
    Examples: UK's GDP per capita between 1990-2020, Japan's Population in 
    1970-2010 snd so on.
    
    Args:
        country (str): The country. e.g. Italy.
        field (str): The desired data about the country. e.g. GDP, GDP Per Capita, Population, Growth-Rate, ...
        years (list[int]): A list of the relevant years.
    '''
    
    # Fetch the country code and verify that it is legal:
    country_code = wb.economy.coder(country)
    if country_code == None:
        errMsg = f"ERROR: Couldn\'t find \"{country}\" in WBGAPI data base."
        print(errMsg)
        return errMsg
    
    # Fetch the field and verify that it is legal:
    field_search_str = str(wb.search(field))
    if field_search_str.find('Series: ') == -1:
        errMsg = f"ERROR: Couldn\'t find \"{field}\" in WBGAPI data base."
        print(errMsg)
        return errMsg
    field_code = field_search_str.split('Series: ')[1].split('\n')[0]
    
    # Get the information from WB and returns it as a list in the format:
    # ['country', data_year1, data_year2, ...]. If there are several countries 
    # then returns a 2D list: [[list for country1], [list for country2], ...]
    wb_data = wb.data.DataFrame(field_code, country_code, years)
    result = [country] + list(wb_data.T[country_code])
    # 2D case:
    # result = [[country[i]] + list(wb_data.T[country_code[i]]) for i in range(country_code)]  # 2D case (several countries).
    return result

# Plot a single curve on a single graph (figure):
def plot_graph(y: list[float], x: list[int], title: str, xlabel: str, ylabel: str):
    '''
    Plots a single y(x) curve graph of the given data.
    
    Args:
        y (list[float]): A list of the country's data points (y axis).
        x (list[int]): A list of the relevant years (x axis).
        title (str): A short title for the graph.
        xlabel (str): "name [units]" for the x-axis.
        ylabel (str): "name [units]" for the y-axis.
    '''
    fig = plt.figure()
    fig.tight_layout()
    plt.plot(x, y, '*-')
    plt.xticks(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
    return "Success."

# Plot multiple curves on a single graph (figure):
def plot_multi_graph(y: list[list[float]], x: list[int], title: str, xlabel: str, ylabel: str, leged_labels: list[str] = None):
    '''
    Plots a graph of several y(x) curves of the given data (y1(x), y2(x), ...).
    
    Args:
        y (list[list[float]]): A list of lists of data points of a country(ies) (y1, y2, ...). Example: y=[y1, y2] where y1 is USA's GDP and y2 is UK's GDP, both by year.
        x (list[int]): A list of the relevant years (x axis). Example: a set of years.
        title (str): A short title for the graph.
        xlabel (str): "name [units]" for the x-axis.
        ylabel (str): "name [units]" for the y-axis.
        leged_labels (list[str]): a very short label-name for each curve. Example: If y1 is Norway's population and y2 is China's army size, then leged_labels=["Norway pop.", "Chinese army"]
    '''
    fig = plt.figure()
    fig.tight_layout()
    for i in range(len(y)):
        plt.plot(x, y[i], '*-', label=(leged_labels[i] if leged_labels != None else str(i)))
    plt.xticks(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.show()
    
    return "Success."

def plot_bars(data_by_entity: list[list], title: str, xlabel: str, ylabel: str, leged_labels: list[str] = None):
    '''
    Plots a bar graph of value(s) per entity.
    
    Args:
        data_by_entity (list[list]): A list of lists. The first value of each list is the entity's name (e.g a country) and the rest are its values (i.e. several bars for each entity).
        title (str): A short title for the graph.
        xlabel (str): Name for the x-axis. e.g. "Countries"
        ylabel (str): "name [units]" for the y-axis.
        leged_labels (list[str]): A very short label-name for each bar/value of an entity (e.g ["year 1", "year 2"]).
    '''
    
    # Extract country names and values
    countries = [entry[0] for entry in data_by_entity]
    values = [entry[1:] for entry in data_by_entity]
    
    # Number of values for each country
    num_values = len(values[0])
    
    # Width of each bar
    bar_width = 1 / (num_values + 2)
    
    # Calculate the x positions for bars
    x = range(len(countries))
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Plot bars for each set of values with some spacing
    for i in range(num_values):
        x_pos = [pos + i * bar_width for pos in x]
        ax.bar(x_pos, [v[i] for v in values], width=bar_width, label=f"Value {i+1}")
    
    # Set x-axis labels
    ax.set_xticks([pos + 0.5 * bar_width for pos in x])
    ax.set_xticklabels(countries)
    ax.set_xlabel(xlabel)
    
    # Set y-axis label
    ax.set_ylabel(ylabel)
    
    # Add a legend for the values
    if leged_labels != None:
        ax.legend(leged_labels)
    
    # Set the title
    plt.title(title)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Normalizes the data list by the population list (i.e. returns the budget-per-person per year):
def normalize_data_list_per_person(data: list[float], population: list[int]):
    '''
    Gets 2 lists: Data values and population values.
    Returns a list of the data, when each value is normalized/divided by its 
    relevant population value.
    
    Args:
        data (list[float]): A list data values. Example: A country's budget per year.
        population (list[int]): A list of the assosiated populations. Example: The country's population per year.
    '''
    if len(data) != len(population) or len(data) == 0:
        return "failed! The given arguments must have equal and positive lengths!"
    return [data[i] / population[i] for i in range(len(data))]

# Gets a variable type (as a string) in a python style, and returns its json 
# format (as a dict). Examples: 
#   "str"  --> {'type': 'string'}
#   "int"  --> {'type': 'integer'}
#   "list[float]" --> {'type': 'array', 'items': {'type': 'number'}}
#   "list[list[float]]" --> {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'number'}}}
def convert_python_types_to_json(type_str):
    basic_types_map = {"str": "string", "int": "integer", "float": "number"}
    array_types_map = {"tuple": "array", "list": "array"}
    
    type_str = type_str.split("'")[1] if type_str.find("'") >= 0 else type_str
    openBracket_i = type_str.find('[')
    if openBracket_i > 0 and type_str[:openBracket_i] in array_types_map:
        arr_type = array_types_map[type_str[:openBracket_i]]
        item_type = convert_python_types_to_json(type_str[openBracket_i + 1:-1])
        return {"type": arr_type, "items": item_type}
    if type_str in basic_types_map:
        return {"type": basic_types_map[type_str]}
    return {"type": "object"}

# Generates a prompt text to understand and operate the given function, for the 
# "gpt-3.5-turbo" model function calling:
def generate_function_prompt(func):
    func_info = inspect.getfullargspec(func)  # meta_data object of the function.
    
    # Prepare the informaton about each argument in the function:
    required_arguments = list(argName for argName, argProps in \
                inspect.signature(func).parameters.items() if argProps.default \
                    is inspect.Parameter.empty)  # Works fine even with just [].
    argument_properties_dict = {}
    for argName in func_info.args:
        argument_properties_dict[argName] = {
            "description": func.__doc__.split(argName + " (")[-1].split(":")[1].split("\n")[0]
            } | convert_python_types_to_json(str(func_info.annotations[argName]))  # The datatype.
    
    json_prompt = {"name": func.__name__,
                   "description": func.__doc__.split('\n    \n')[0],
                    "parameters": {
                        "type": "object",
                        "properties": argument_properties_dict,
                        "required": required_arguments  # Works fine even with just [].
                        }
                    }
    return json_prompt


# Run the GPT model for the given prompt:
def run_model(prompt_tup):
    available_functions = {"get_world_bank_data": get_world_bank_data, 
                           "plot_graph": plot_graph, "plot_multi_graph": plot_multi_graph, 
                           "normalize_data_list_per_person": normalize_data_list_per_person,
                           "plot_bars": plot_bars}
    model = "gpt-3.5-turbo"
    system_content, user_content = prompt_tup[0], prompt_tup[1]
    messages = [{"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
                ]
    functions = [generate_function_prompt(func) for func in available_functions.values()]

    # Send the prompt message:
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto"
        )
    response_message = response["choices"][0]["message"]

    while response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        # Verify that the function exists:
        if function_name not in available_functions:
            print('Error: The model trie to call a function that doesn\'t exists!\n')
            break
        fuction_to_call = available_functions[function_name]
        function_args = list(json.loads(response_message["function_call"]["arguments"]).values())
        # # Verify that the types of the arguments are valid:
        # func_info = inspect.getfullargspec(fuction_to_call)
        # ....
        function_response = fuction_to_call(*function_args)
        
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        )  # extend conversation with function response
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
        )
        response_message = response["choices"][0]["message"]

    # No more functions to call, meaning this is the final answer:
    print('Response:\n', response_message['content'])

#%%

# Try running the code:

system_role_prompt = "You're an analyst in an international investment company."

# user_request_prompt = "What is the GDP per capita PPP of the Netherlands in the years 2012-2022? Plot the results in a graphs."
# user_request_prompt = "What is the GDP per capita PPP of the Netherlands and UK in the years 2012-2022? Plot the results in a single graphs."
# user_request_prompt = "What is the GDP per capita PPP of the Netherlands and UK in the years 2012-2022? Plot the results in 2 separated graphs."

user_request_prompt = "What is the GDP per capita PPP of Israel, France, UK, Germany, Netherlands, Canada and Australia, in the years 1995-2020? Plot the results in a single graphs."
# user_request_prompt = "What is the \"income share held by highest 10%\" of Israel, France, UK, Germany, Netherlands, Canada and Australia, in the years 1995-2020? Plot the results in a single graphs."
# user_request_prompt = "What is the \"foreign direct investment, net inflows\" in USD, in Israel, France, UK, Germany, Netherlands, Canada and Australia, in the years 1995-2020? Plot the results in a single graphs."
# user_request_prompt = "What is the \"foreign direct investment, net inflows\" in USD, in Israel, France, UK, Germany, Netherlands, Canada and Australia, normalized by each country's local population, in the years 1995-2020? Plot the results in a single graphs."
# user_request_prompt = "What is the exports of goods and services (current US$) of Israel, France, UK, Germany, Netherlands, Canada and Australia, in the years 1995-2020? Plot the results in a single graphs."
# user_request_prompt = "What is the exports of goods and services (current US$) of Israel, France, UK, Germany, Netherlands, Canada and Australia, normalized by each country's local population, in the years 1995-2020? Plot the results in a single graphs."

# user_request_prompt = "What is the median salary in Israel, France, UK, Germany, Netherlands, Canada and Australia, in the years 1995-2020? Plot the results in a single graphs."
# user_request_prompt = "What is the total government budget (current US$) of Israel, France, UK, Germany, Netherlands, Canada and Australia, normalized by each country's local population, in the years 1995-2020? Plot the results in a single graphs."

run_model((system_role_prompt, user_request_prompt))

