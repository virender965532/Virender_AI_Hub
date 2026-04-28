async function start() {
    const res = await fetch("/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            role: document.getElementById("role").value,
            difficulty: document.getElementById("difficulty").value,
            num_questions: document.getElementById("num").value
        })
    });

    const data = await res.json();

    document.getElementById("chat").innerHTML =
        "<b>Q:</b> " + data.current_question;
}


async function submitAnswer() {
    const answer = document.getElementById("answer").value;

    const res = await fetch("/answer", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({answer})
    });

    const data = await res.json();

    document.getElementById("chat").innerHTML +=
        "<br><b>Feedback:</b> " + data.feedback +
        "<br><b>Next Q:</b> " + data.current_question;
}