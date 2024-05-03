import os
from re import template
import sys
import getpass
from typing import Tuple
from paramiko import SSHClient
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.agents import AgentExecutor
from langchain.schema.messages import HumanMessage, AIMessage
from termcolor import colored

chat_history = []

#llm = OpenAI(model_name='text-davinci-003', max_tokens=1000)
#llm = ChatOpenAI(model_name='gpt-4', max_tokens=5000)
llm35 = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.0)
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.0)

ssh = client = SSHClient()
ssh.load_system_host_keys()
# get server info from cli args
user, server = sys.argv[1].split("@")
ssh.connect(server, username=user)


def summarize_std(out: str, source: str) -> str:
    """Uses llm to summarize stdout"""
    nt = llm.get_num_tokens(out)
    if nt > 3000:
        # TODO handle really long output, truncate them
        template = """
        Cleanup following {source} stream, extract only relevant pieces of information, omit the rest with ...:

        {out}

        Relevant information:
        """
        prompt = PromptTemplate(template=template,
                                input_variables=["out", "source"])
        chain = LLMChain(llm=llm35, prompt=prompt)
        summary = chain.run({'out': out, 'source': source})
        return summary
    return out


sudo_password = getpass.getpass("sudo password: ")


@tool
def run_shell_command(command: str) -> Tuple[int, str, str]:
    """Run shell command on remote system, returns exit code, stdout, stderr."""
    global sudo_password
    print(colored("$ " + command, "yellow"))
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    if command.startswith('sudo'):
        stdin.write(sudo_password + '\n')
        stdin.flush()
    exit_code = stdout.channel.recv_exit_status()
    out = summarize_std(stdout.read().decode('utf-8'), "stdout")
    err = summarize_std(stderr.read().decode('utf-8'), "stderr")
    if len(out) > 0:
        print(out.strip())
    if len(err.strip()) > 0:
        print(colored(err.strip(), "red"))
    return exit_code, out, err


@tool
def run_python_script(script: str) -> Tuple[int, str, str]:
    """Run python script on remote system, returns exit code, stdout, stderr."""
    return run_shell_command("python3 {}".format(script))


@tool
def ask_user_a_question(question: str) -> str:
    """Ask user a question."""
    return input(question + " ")


@tool
def ask_yes_or_no(question: str) -> bool:
    """Ask user a yes or no question."""
    return input(question + " ").lower().startswith('y')


tools = [
    run_python_script, run_shell_command, ask_user_a_question, ask_yes_or_no
]

system_prompt = """
You are a powerful interactive AI tool created to configure remote servers via ssh session.
You can run commands on server or ask user for an input.
Be smart about commands you are generating, minimize output coming from the server (to minimize token usage), generate python or bash scripts to be executed on remote server to get answers about environment you are working on.
All generated commands should be non-interactive, not requiring any input from user (use -y flag for apt or apt-get).
Run all apt or apt-get commands as root (use sudo).
Always prefer running a script over asking user a question.
You begin by checking what is installed on it and what distribution it is running.
Stop only when user tells you that job is done.

"""

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools])

agent = {
    "input":
    lambda x: x["input"],
    "agent_scratchpad":
    lambda x: format_to_openai_functions(x['intermediate_steps']),
    "chat_history":
    lambda x: x["chat_history"]
} | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


def main():
    while True:
        ui = input(colored(">>> ", "green"))
        if ui == "exit":
            break
        user_input = "User input: {}".format(ui)
        chat_history.append(HumanMessage(content=user_input))
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        output = result['output']
        print(colored(output, "blue"))
        chat_history.append(AIMessage(content=output))
    ssh.close()


if __name__ == '__main__':
    main()
