import os
import sys
import subprocess
import venv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def create_virtual_environment():
    venv_dir = os.path.join(os.getcwd(), "ai_agents_env")
    venv.create(venv_dir, with_pip=True)
    return venv_dir

def get_venv_python(venv_dir):
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")

def install_dependencies(venv_python):
    subprocess.check_call([venv_python, "-m", "pip", "install", "flask", "flask-cors", "crewai", "langchain", "ollama"])

def setup_environment():
    venv_dir = create_virtual_environment()
    venv_python = get_venv_python(venv_dir)
    install_dependencies(venv_python)
    return venv_python

def run_app(venv_python):
    subprocess.check_call([venv_python, __file__, "run_server"])

class AISystem:
    def __init__(self):
        from langchain.llms import Ollama
        from crewai import Agent, Task, Crew, Process

        llm = Ollama(model="llama2:1b")

        self.ceo = Agent(
            role='CEO',
            goal='Analyze user requests and delegate tasks',
            backstory='You are the CEO of an AI company, responsible for understanding user needs and coordinating the team.',
            verbose=True,
            allow_delegation=True,
            llm=llm
        )
        
        self.manager = Agent(
            role='Manager',
            goal='Coordinate tasks and oversee their execution',
            backstory='You are a skilled project manager, responsible for breaking down tasks and ensuring their completion.',
            verbose=True,
            allow_delegation=True,
            llm=llm
        )
        
        self.researcher = Agent(
            role='Researcher',
            goal='Gather and analyze information from various sources',
            backstory='You are an expert at finding and synthesizing information from the internet and other sources.',
            verbose=True,
            llm=llm
        )
        
        self.writer = Agent(
            role='Writer',
            goal='Create high-quality written content',
            backstory='You are a skilled writer, capable of producing engaging and informative content on various topics.',
            verbose=True,
            llm=llm
        )

    def process_request(self, user_request):
        from crewai import Task, Crew, Process

        analyze_task = Task(
            description=f"Analyze the following user request and determine the necessary steps: {user_request}",
            agent=self.ceo
        )

        plan_task = Task(
            description="Create a detailed plan to fulfill the user request based on the CEO's analysis",
            agent=self.manager
        )

        research_task = Task(
            description="Conduct necessary research to support the plan",
            agent=self.researcher
        )

        execute_task = Task(
            description="Execute the plan and produce the required output",
            agent=self.writer
        )

        crew = Crew(
            agents=[self.ceo, self.manager, self.researcher, self.writer],
            tasks=[analyze_task, plan_task, research_task, execute_task],
            verbose=2,
            process=Process.sequential
        )

        result = crew.kickoff()
        return result

ai_system = None

@app.route('/chat', methods=['POST'])
def chat():
    global ai_system
    if ai_system is None:
        ai_system = AISystem()
    data = request.json
    user_message = data['message']
    response = ai_system.process_request(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_server":
        app.run(port=5000)
    else:
        venv_python = setup_environment()
        run_app(venv_python)