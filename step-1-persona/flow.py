from agent import Agent

if __name__ == "__main__":
    # Create a financial advisor agent
    financial_advisor = Agent("StudentAdvisor")
    
    financial_advisor.persona = """My son going to college this year as computer science student and I want  him to be involve in any program like Nvidia university Ambdassador program."""
    
    # Execute a task
    task = "Advise him what he should do to be part of the program and what are the benefits of the program."
    response = financial_advisor.execute(task)
    
    print(f"Agent Name: {financial_advisor.name}")
    print(f"Agent Persona: {financial_advisor.persona}")
    print(f"\nTask: {task}")
    print(f"Agent Response:\n{response}")
 
    # Execute another task with the same persona
    task = "Explain the steps which course he should take to be part of the program and what certificate he will get after completion of the program."
    response = financial_advisor.execute(task)
    
    print(f"\nNew Task: {task}")
    print(f"Agent Response:\n{response}")