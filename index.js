const columns = [
  "golden_question",
  "full_answer",
  "golden_answer",
  "is_correct",
];

const title_map = {
  golden_question: "Question",
  full_answer: "Generated Answer",
  golden_answer: "Correct Answer",
  is_correct: "Prediction",
};

const get_compare_data = async () => {
  const response1 = await fetch("results/gsm8k_gpt-3.5-turbo-0301_cot.jsonl");
  const response2 = await fetch(
    "results/gsm8k_gpt-3.5-turbo-0301_cot_rephrase_v1.jsonl"
  );
  const text1 = await response1.text();
  const text2 = await response2.text();
  const data1 = parseData(text1);
  const data2 = parseData(text2);
  const randomIndex = Math.floor(Math.random() * data1.length);
  const original = data1.find((item) => item.id === randomIndex);
  const rephrased = data2.find((item) => item.id === randomIndex);
  return { original, rephrased };
};

const parseData = (text) => {
  return text
    .split("\n")
    .map((line) => {
      let q = null;
      try {
        q = JSON.parse(line);
      } catch (e) {
        console.log(e);
      }
      return q;
    })
    .filter((line) => line !== null);
};

async function showRandomExample() {
  const table = document.getElementById("myTable");
  const data = await get_compare_data();
  const original = data.original;
  const rephrased = data.rephrased;

  // Clear the table
  table.innerHTML = "";

  // Add table headers
  const headerRow = table.insertRow();
  columns.forEach((key) => {
    const headerCell = document.createElement("th");
    headerCell.textContent = title_map[key];
    headerRow.appendChild(headerCell);
  });

  // Add original data row
  const originalRow = table.insertRow();
  columns.forEach((key) => {
    const newCell = originalRow.insertCell();
    if (key === "is_correct") {
      newCell.innerHTML = original[key] ? "&#10004;" : "&#10006;";
    } else {
      newCell.textContent = original[key];
    }
  });

  // Add rephrased data row
  const rephrasedRow = table.insertRow();
  columns.forEach((key) => {
    const newCell = rephrasedRow.insertCell();
    if (key === "is_correct") {
      newCell.innerHTML = rephrased[key] ? "&#10004;" : "&#10006;";
    } else {
      newCell.textContent = rephrased[key];
    }
  });
}
