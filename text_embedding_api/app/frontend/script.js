async function getEmbedding(text) {
    const url = "http://127.0.0.1:8000/embed-text"
    const body = {"texts": [text]}

    try {
        const response = await fetch(
            url,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(body)
            }
        )

        if (!response.ok) {
            throw new Error("Network response was not ok ", response);
        }

        const result = await response.json()
        console.log("Response from server:", result)
        return result['data'][0]['embeddings']
    } 
    catch (error) {
        console.error("Error posting JSON:", error)
    }
}

function loadInputAndConvert() {
    const form = document.getElementById("form");
    const input = document.getElementById("original-text");
    const output = document.getElementById("embedded-text");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        output.value = await getEmbedding(input.value)
    });
}

loadInputAndConvert();