import requests
import json
import base64


def get_github_content(owner, repo, path, token):
    """
    Fetch a list of files or content of a specific file from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_file_content(owner, repo, path, token):
    """
    Fetch the raw content of a file from a GitHub repository, assuming the file is JSON.
    """
    content = get_github_content(owner, repo, path, token)
    if 'content' in content:
        # Decode Base64 content and load it as JSON
        raw_content = json.loads(base64.b64decode(content['content']).decode('utf-8'))
        return raw_content
    return {}


def detect_framework_from_files(files, owner, repo, token):
    """
    Detect testing frameworks based on filenames and content in the repository.
    """
    framework_patterns = {
        "pytest": ["pytest.ini", "conftest.py", "pyproject.toml"],
        "unittest": [],
        "Jest": ["jest.config.js", "package.json"],
        "Mocha": ["mocha.opts", "package.json", ".mocharc.json"],
        "Karma": ["karma.conf.js"],
        "JUnit": ["pom.xml", "build.gradle"],
        "TestNG": ["testng.xml"],
        "NUnit": ["nunit.config", "*.csproj"],
        "xUnit": ["*.csproj"],
        "PHPUnit": ["phpunit.xml", "phpunit.xml.dist"],
        "RSpec": [".rspec", "spec_helper.rb", "rails_helper.rb"],
        "Minitest": ["minitest_helper.rb"],
        "Google Test": [],
        "Boost.Test": [],
        "Go test": ["go.mod"],
        "XCTest": ["*.xcodeproj"]
    }

    detected_frameworks = set()

    for framework, patterns in framework_patterns.items():
        for pattern in patterns:
            if any(pattern in file['name'] for file in files):
                if pattern == "package.json":
                    # Fetch and parse the content of package.json
                    path = [file['path'] for file in files if file['name'] == 'package.json'][0]
                    file_content = get_file_content(owner, repo, path, token)
                    if 'jest' in file_content.get('devDependencies', {}):
                        detected_frameworks.add("Jest")
                    if 'mocha' in file_content.get('devDependencies', {}):
                        detected_frameworks.add("Mocha")
                else:
                    detected_frameworks.add(framework)

    return detected_frameworks if detected_frameworks else None


def main():
    owner = "costrojs"
    repo = "costro"
    token = "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"

    try:
        files = get_github_content(owner, repo, '', token)
        frameworks = detect_framework_from_files(files, owner, repo, token)
        if frameworks:
            print("Detected Frameworks:", ", ".join(frameworks))
        else:
            print("Detected Frameworks: None")
    except requests.HTTPError as e:
        print("HTTP Error:", e)
    except requests.RequestException as e:
        print("Error fetching data from GitHub:", e)


if __name__ == "__main__":
    main()
