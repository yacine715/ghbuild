import requests
import re
import zipfile
import io
import base64

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
        'rspec': ['Gemfile' , 'Rakefile'],
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
        'rspec': re.compile(r'rspec'),   #r'gem\s*[\'"]rspec[\'"]|require\s*[\'"]rspec[\'"]'),
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
                    print("Framework: ", framework)
                    #print("Content: ", content)
                    if framework_dependencies[framework].search(content):
                        frameworks_found.append(framework)
                except Exception as e:
                    continue
    
    return frameworks_found

framework_regex = {
    "pytest": re.compile(r"(?:(\d+)\s+passed)?(?:, )?(?:(\d+)\s+failed)?(?:, )?(?:(\d+)\s+skipped)?"),
    "Jest": re.compile(r"Tests: (\d+) total, (\d+) passed, (\d+) failed, (\d+) skipped"),
    "junit": re.compile(r"Passed: (\d+), Failed: (\d+), Errors: (\d+), Skipped: (\d+)"),
    "rspec": re.compile(r"(\d+) examples?, (\d+) failures?(?:, (\d+) pending)?"),
    "PHPUnit": re.compile(r"Tests: (\d+), Assertions: (\d+), Failures: (\d+), Skipped: (\d+)"),
    "NUnit": re.compile(r"Total tests: (\d+) - Passed: (\d+), Failed: (\d+), Skipped: (\d+)"),
    "Go test": re.compile(r"PASS: (\d+), FAIL: (\d+), SKIP: (\d+)"),
    "Maven": re.compile(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)"),
    "cucumber-ruby": re.compile(r"(\d+) scenarios? \((?:(\d+ skipped)(?:, )?)?(?:(\d+ undefined)(?:, )?)?(?:(\d+ failed)(?:, )?)?(?:(\d+ passed))?\)[\s\S]*?(\d+) steps? \((?:(\d+ skipped)(?:, )?)?(?:(\d+ undefined)(?:, )?)?(?:(\d+ failed)(?:, )?)?(?:(\d+ passed))?\)"),
    "Cucumber-Java": re.compile(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)"),
    "testunit": re.compile(r"(\d+) tests, (\d+) assertions, (\d+) failures, (\d+) errors, (\d+) pendings, (\d+) omissions, (\d+) notifications")
}

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def get_github_repo_files(owner, repo, token=None):
    """
    Fetch the list of files in the root of a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    headers = {"Authorization": f"token {token}"} if token else {}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return [file['name'] for file in response.json() if file['type'] == 'file']

def get_github_actions_runs(owner, repo, token=None):
    """
    Fetch the list of workflow runs for a GitHub repository.
    """
    #fourth page of runs

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    headers = {"Authorization": f"token {token}"} if token else {}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    #length of runs
    print("Length of runs: ", len(response.json()['workflow_runs']))
    return response.json()['workflow_runs']

def get_github_actions_log(owner, repo, run_id, token=None):
    """
    Fetch the logs for a specific GitHub Actions workflow run.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    headers = {"Authorization": f"token {token}"} if token else {}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.content

def parse_test_results(framework, log_content):
    """
    Parse the test results from the log content.
    """
    log_content = remove_ansi_escape_sequences(log_content)  # Remove ANSI escape codes

    if framework in framework_regex:
        regex = framework_regex[framework]
        matches = regex.findall(log_content)
        
        # Debug statements
        #print("Log content: ", log_content)
        #print("Matches found: ", matches)
        
        if matches:
            passed_tests = 0
            failed_tests = 0
            skipped_tests = 0
            errors_tests = 0

            for match in matches:
                if framework == "pytest":
                    if match[0]:
                        passed_tests += int(match[0])
                    if match[1]:
                        failed_tests += int(match[1])
                    if match[2]:
                        skipped_tests += int(match[2])
                elif framework == "junit":
                    passed_tests += int(match[0])
                    failed_tests += int(match[1])
                    errors_tests += int(match[2])  # Count errors for JUnit
                    skipped_tests += int(match[3])

                elif framework == "rspec":
                    if match[0]:
                        passed_tests += int(match[0])
                    if match[1]:
                        failed_tests += int(match[1])
                    if match[2]:
                        skipped_tests += int(match[2])
                elif framework == "cucumber-ruby":
                    scenarios_skipped = int(match[1].split()[0]) if match[1] else 0
                    scenarios_undefined = int(match[2].split()[0]) if match[2] else 0
                    scenarios_failed = int(match[3].split()[0]) if match[3] else 0
                    scenarios_passed = int(match[4].split()[0]) if match[4] else 0
                    steps_skipped = int(match[6].split()[0]) if match[6] else 0
                    steps_undefined = int(match[7].split()[0]) if match[7] else 0
                    steps_failed = int(match[8].split()[0]) if match[8] else 0
                    steps_passed = int(match[9].split()[0]) if match[9] else 0
                    
                    passed_tests += scenarios_passed + steps_passed
                    failed_tests += scenarios_failed + steps_failed
                    skipped_tests += scenarios_skipped + steps_skipped
                    #undefined_tests += scenarios_undefined + steps_undefined
                    # No skipped or errors for this format
                    # No errors for this format
                elif framework == "Cucumber-Java":
                    passed_tests += int(match[0])
                    failed_tests += int(match[1])
                    errors_tests += int(match[2])
                    skipped_tests += int(match[3])
                elif framework == "testunit":
                    passed_tests += int(match[0])
                    #assertions += int(match[1])
                    failed_tests += int(match[2])
                    errors_tests += int(match[3])
                    # pendings, omissions, and notifications are not being counted in total

            total_tests = passed_tests + failed_tests + skipped_tests  + errors_tests

            return {
                'passed': passed_tests,
                'failed': failed_tests,
                'skipped': skipped_tests,
                'total': total_tests
            }
    
    return {'passed': 0, 'failed': 0, 'skipped': 0, 'total': 0}

def summarize_test_results(test_results):
    """
    Summarize the test results in the desired format.
    """
    summary = {
        'tr tests ok': test_results.get('passed', 0),
        'tr tests fail': test_results.get('failed', 0),
        'tr tests run': test_results.get('total', 0),
        'tr tests skipped': test_results.get('skipped', 0),
        'tr failed tests': []  # Optionally add the names of failed tests if available
    }
    return summary

def main():
    owner = "stripe"
    repo = "stripe-ruby"
    token = "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"  # replace with your GitHub token

    # Fetch the latest workflow run and its logs
    runs = get_github_actions_runs(owner, repo, token)
    latest_run_id = runs[1]['id']  # Note: Adjust index if necessary
    log = get_github_actions_log(owner, repo, latest_run_id, token)

    # Save the log file
    with open('actions_log.zip', 'wb') as f:
        f.write(log)

    print("Log downloaded for run ID:", latest_run_id)

    # Identify test framework
    repo_files = get_github_repo_files(owner, repo, token)
    frameworks = identify_test_frameworks(repo_files, owner, repo, token)
    print("Frameworks found:", frameworks)
    detected_framework = frameworks[0] if frameworks else "Unknown"
    print(f"Detected test framework: {detected_framework}")

    # Initialize cumulative test results
    cumulative_test_results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'total': 0
    }

    # Parse the test results from the log content
    with zipfile.ZipFile(io.BytesIO(log), 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.txt'):  # Adjust the extension as needed
                with zip_ref.open(file_info) as log_file:
                    log_content = log_file.read().decode('utf-8')
                    test_results = parse_test_results(detected_framework, log_content)
                    # Accumulate test results
                    cumulative_test_results['passed'] += test_results['passed']
                    cumulative_test_results['failed'] += test_results['failed']
                    cumulative_test_results['skipped'] += test_results['skipped']
                    cumulative_test_results['total'] += test_results['total']
                    print(f"Parsed test results from {file_info.filename}: {test_results}")
                    print

    # Summarize the test results
    summary = summarize_test_results(cumulative_test_results)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
