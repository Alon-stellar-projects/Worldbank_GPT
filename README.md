# Worldbank_GPT
Author: Alon Haviv

This Python program lets a user extract, analyze and display data from the World-Bank API, using GPT and openAI function-calling.

Usage:

* Open "worldbank_gpt.py" file. In it set the "openai.api_key" value to your own key (str).
* On a new python file, import the content of "worldbank_gpt.py".
* After importing the code, run the following line (Python):
    run_model((system_role_prompt, user_request_prompt))
  Where "system_role_prompt" and "user_request_prompt" are prompt strings for the GPT model. Your terminology must match World-Bank's database terminology, which sometimes is hard.
  Example:
    system_role_prompt = "You're an analyst."
    user_request_prompt = "What is the GDP per capita PPP of the Netherlands in the years 2012-2022? Plot the results in a graphs."

Files:
1) README:
   This file.
2) worldbank_gpt.py:
   Contains the functionality.
Additional image file(s) to show examples of results.
