
import os
from getpass import getpass

from crewai import Agent, Crew, Process, Task, LLM

# --------------------------------
# 1. Configure Gemini API Key
# --------------------------------

os.environ["GEMINI_API_KEY"] = getpass("Enter your Gemini API key: ")

# --------------------------------
# 2. Configure Gemini LLM
# --------------------------------

llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

# --------------------------------
# 3. Create Agents
# --------------------------------

researcher = Agent(
    role="Researcher",
    goal="Gather accurate, relevant information and key talking points on the given topic",
    backstory="You are a diligent researcher who finds facts, context, and angles readers care about.",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Draft clear, engaging blog posts grounded in the research provided",
    backstory="You are a skilled blogger who explains ideas in plain language with a strong narrative arc.",
    llm=llm,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Improve drafts for clarity, flow, grammar, and engagement without changing the core message",
    backstory="You are an experienced editor who tightens prose and sharpens hooks and conclusions.",
    llm=llm,
    verbose=True
)

# --------------------------------
# 4. Create Tasks
# --------------------------------

research_task = Task(
    description="Research the topic: {topic}. Identify main themes, facts, and angles worth covering.",
    expected_output="A structured summary of findings (bullet points or short sections) the writer can use.",
    agent=researcher
)

write_task = Task(
    description="Using the research, write a blog post about {topic} of roughly 500 words. Include introduction, body, and conclusion.",
    expected_output="A complete 500-word blog draft in markdown.",
    agent=writer,
    context=[research_task]
)

edit_task = Task(
    description="Edit the draft for clarity, tone, and engagement. Fix awkward phrasing and strengthen opening and closing.",
    expected_output="The final polished blog post, ready to publish.",
    agent=editor,
    context=[write_task]
)

# --------------------------------
# 5. Create Crew
# --------------------------------

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
    verbose=True
)

# --------------------------------
# 6. Execute Crew Asynchronously
# --------------------------------

result = await crew.kickoff_async(
    inputs={
        "topic": "Why multi-agent systems matter for product teams"
    }
)

# --------------------------------
# 7. Display Final Result
# --------------------------------

print("\n\n========== FINAL BLOG POST ==========\n")
print(result)
