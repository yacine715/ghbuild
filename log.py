import os
import requests
import re
import zipfile
import io


def get_github_actions_runs(owner, repo, token=None):

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['workflow_runs']


def get_github_actions_log(owner, repo, run_id, token=None):

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.content


# Regex patterns to capture test results from the log content
test_summary_pattern = re.compile(r"Ran (\d+) tests in [\d\.]+s")
test_outcome_pattern = re.compile(r"OK \(skipped=(\d+)\)")


def parse_test_results(log_content):

    results = {'passed': 0, 'failed': 0, 'skipped': 0, 'total': 0}
    summary_match = test_summary_pattern.search(log_content)
    outcome_match = test_outcome_pattern.search(log_content)

    if summary_match:
        results['total'] = int(summary_match.group(1))
    if outcome_match:
        results['skipped'] = int(outcome_match.group(1))
        results['passed'] = results['total'] - results['skipped']  # Assuming all non-skipped tests passed

    return results


def main():
    """
    Main function to orchestrate fetching, parsing, and summarizing test results.
    """
    owner = "quarkiverse"
    repo = "quarkus-jberet"
    token = "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"

    # Fetch the latest workflow run ID
    runs = get_github_actions_runs(owner, repo, token)
    if not runs:
        print("No runs found for the specified repository.")
        return
    latest_run_id = runs[0]['id']
    print("Log downloaded for run ID:", latest_run_id)

    # Download the log file
    log = get_github_actions_log(owner, repo, latest_run_id, token)

    # Save the log file locally
    with open('actions_log.zip', 'wb') as f:
        f.write(log)

    # Initialize cumulative results
    cumulative_results = {'passed': 0, 'failed': 0, 'skipped': 0, 'total': 0}

    # Parse each log file within the downloaded zip archive
    with zipfile.ZipFile(io.BytesIO(log), 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.txt'):
                with zip_ref.open(file_info) as log_file:
                    log_content = log_file.read().decode('utf-8')
                    test_results = parse_test_results(log_content)
                    cumulative_results['passed'] += test_results['passed']
                    cumulative_results['failed'] += test_results['failed']
                    cumulative_results['skipped'] += test_results['skipped']
                    cumulative_results['total'] += test_results['total']
                    print(f"Parsed test results from {file_info.filename}: {test_results}")

    # Output the cumulative results
    print("Cumulative Results:", cumulative_results)


if __name__ == "__main__":
    main()
