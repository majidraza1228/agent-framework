# test_tools.py
import os
from agent import Agent
from wikipedia_tool import WikipediaTool
from websearch_tool import WebSearchTool

def main():
    # Check for API keys
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    if not os.getenv('TAVILY_API_KEY'):
        print("Please set your TAVILY_API_KEY environment variable")
        return
    
    try:
        # Initialize agent and tools
        agent = Agent("research_agent")
        wiki_tool = WikipediaTool()
        web_tool = WebSearchTool()
        
        # Register both tools
        agent.register_tool(wiki_tool)
        agent.register_tool(web_tool)
        
        # Set up agent persona
        agent.persona = """I am a research assistant with access to both Wikipedia and web search.
        I can find information from multiple sources to provide comprehensive answers.
        When searching for current developments, I'll use web search first."""
        
        # Test queries
        queries = [
            "What are the latest developments in quantum computing?",
            "Compare classical computers and quantum computers",
            "What is the current state of quantum supremacy?"
        ]
        
        for query in queries:
            print(f"\nExecuting query: {query}")
            response = agent.execute(query)
            print("\nResponse:")
            print(response)
            print("\n" + "="*50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()