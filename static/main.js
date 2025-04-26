document.getElementById('generate').addEventListener('click', async () => {
    const prefix = document.getElementById('prefix').value;
    const spinner = document.getElementById('spinner');
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '';
    spinner.style.display = 'inline-block';
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prefix})
        });
        const data = await response.json();
        spinner.style.display = 'none';
        resultDiv.innerHTML = `
            <p><strong>Public Key:</strong> ${data.public_key}</p>
            <p><strong>Secret Key:</strong> ${data.secret_key}</p>
            <p><strong>Tries:</strong> ${data.tries}</p>
            <p><strong>Elapsed Time:</strong> ${data.elapsed.toFixed(2)}s</p>
        `;
    } catch (e) {
        spinner.style.display = 'none';
        resultDiv.innerHTML = `<p class="text-danger">Error generating wallet</p>`;
    }
});
