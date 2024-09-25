import requests
import re
import pandas as pd
import base64

# Define the functions from your previous implementations

def get_github_repo_files(owner, repo, token=None):
    """
    Fetch the list of files in the root of a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    headers = {"Authorization": f"token {token}"} if token else {}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return [file['name'] for file in response.json() if file['type'] == 'file']

def get_file_content(owner, repo, path, token=None):
    """
    Fetch the content of a file from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"} if token else {}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content = response.json().get('content', '')
    return base64.b64decode(content).decode('utf-8')

def identify_test_frameworks(files, owner, repo, token=None):
    """
    Identify the test frameworks based on the presence of specific dependencies in build files.
    """
    test_framework_mapping = {
        'junit': ['pom.xml', 'build.gradle'],
        'testunit': ['Gemfile'],
        'cucumber-ruby': ['Gemfile', 'Rakefile'],
        'cucumber-java': ['pom.xml', 'build.gradle'],
        'phpunit': ['composer.json'],
        'pytest': ['requirements.txt', 'setup.py', 'pyproject.toml'],
        'unittest': ['requirements.txt', 'setup.py', 'pyproject.toml'],
        'jest': ['package.json'],
        'mocha': ['package.json']
    }
    framework_dependencies = {
        'junit': re.compile(r'junit'),
        'testunit': re.compile(r'gem\s*[\'"]test-unit[\'"]'),
        'cucumber-ruby': re.compile(r'gem\s*[\'"]cucumber[\'"]|cucumber'),
        'cucumber-java': re.compile(r'cucumber-java|cucumber-junit|io.cucumber:cucumber'),
        'phpunit': re.compile(r'"phpunit/phpunit"'),
        'pytest': re.compile(r'pytest'),
        'unittest': re.compile(r'unittest'),
        'jest': re.compile(r'"jest"'),
        'mocha': re.compile(r'"mocha"')
    }
    
    frameworks_found = []
    
    for framework, paths in test_framework_mapping.items():
        for path in paths:
            if path in files:
                try:
                    content = get_file_content(owner, repo, path, token)
                    if framework_dependencies[framework].search(content):
                        frameworks_found.append(framework)
                except Exception as e:
                    continue
    
    return frameworks_found

# Define the main function to process the CSV file and add the detected frameworks
def main():
    input_csv = 'projects_cleaned.csv'  # Your input CSV file
    output_csv = 'github_projects_with_frameworks.csv'  # Output CSV file
    token = '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'  # Replace with your GitHub token if necessary

    # Read the CSV file
    df = pd.read_csv(input_csv, header=None, names=['url'])

    # Initialize a new column for the detected frameworks
    df['frameworks'] = ''

    # Process each URL in the CSV
    for index, row in df.iterrows():
        url = row['url']
        owner_repo = url.split('https://github.com/')[-1]
        owner, repo = owner_repo.split('/')
        print(f"Processing {owner}/{repo}...")
        
        try:
            # Get the list of files in the repo
            repo_files = get_github_repo_files(owner, repo, token)
            # Identify the test frameworks
            frameworks = identify_test_frameworks(repo_files, owner, repo, token)
            # Update the DataFrame with the detected frameworks
            df.at[index, 'frameworks'] = ', '.join(frameworks)
        except Exception as e:
            df.at[index, 'frameworks'] = 'Error: ' + str(e)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
