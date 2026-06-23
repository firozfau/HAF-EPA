const recommendations = window.recommendations || [];

document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("recommendationChart");
    const employeeSelect = document.getElementById("employeeSelect");
    const skillSelect = document.getElementById("skillSelect");
    const chartTitle = document.getElementById("chartTitle");

    if (!canvas || !employeeSelect || !skillSelect || !chartTitle) {
        console.error("Graph required HTML elements are missing.");
        return;
    }

    let chartInstance = null;

    function numberValue(value, defaultValue = 0) {
        const numericValue = Number(value);
        return Number.isFinite(numericValue) ? numericValue : defaultValue;
    }

    function getFilteredRecommendations() {
        const selectedEmployee = employeeSelect.value;

        return recommendations.filter(item => {
            return selectedEmployee === "all" || item.full_name === selectedEmployee;
        });
    }

    function getChartData() {
        const filteredData = getFilteredRecommendations();

        return {
            rows: filteredData,
            labels: filteredData.map(item => `#${item.recommendation_rank} ${item.full_name}`),
            managementSkill: filteredData.map(item => numberValue(item.management_score)),
            programmingSkill: filteredData.map(item => numberValue(item.programming_score)),
            projectMatchSkill: filteredData.map(item => numberValue(item.match_percentage)),
            mlSuitability: filteredData.map(item => numberValue(item.ml_prediction_percentage)),
            matchedRequiredSkills: filteredData.map(item => numberValue(item.matched_required_skill_count))
        };
    }

    function destroyOldChart() {
        if (chartInstance) {
            chartInstance.destroy();
            chartInstance = null;
        }
    }

    function updateCards(data) {
        const total = data.rows.length;

        const topMatch = data.projectMatchSkill.length
            ? Math.max(...data.projectMatchSkill)
            : 0;

        const topMlScore = data.mlSuitability.length
            ? Math.max(...data.mlSuitability)
            : 0;

        const averageMatchedSkills = data.matchedRequiredSkills.length
            ? data.matchedRequiredSkills.reduce((a, b) => a + b, 0) / data.matchedRequiredSkills.length
            : 0;

        document.getElementById("totalEmployeesCard").textContent = total;
        document.getElementById("topMatchCard").textContent = `${topMatch.toFixed(2)}%`;
        document.getElementById("mlScoreCard").textContent = `${topMlScore.toFixed(2)}%`;
        document.getElementById("matchedSkillCard").textContent = averageMatchedSkills.toFixed(1);
    }

    function commonOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false
            },
            plugins: {
                legend: {
                    position: "top"
                },
                tooltip: {
                    backgroundColor: "#0b1b3a",
                    titleColor: "#ffffff",
                    bodyColor: "#ffffff",
                    padding: 14,
                    cornerRadius: 8,
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label}: ${Number(context.raw).toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 100,
                    ticks: {
                        callback: function (value) {
                            return value + "%";
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        };
    }

    function getSelectedDatasets(data) {
        const selectedSkill = skillSelect.value;

        if (selectedSkill === "programming") {
            chartTitle.textContent = "Programming Skill vs Project Match";

            return [
                {
                    label: "Programming Skill",
                    data: data.programmingSkill,
                    backgroundColor: "rgba(24, 119, 242, 0.75)",
                    borderColor: "#1877f2",
                    borderWidth: 2,
                    borderRadius: 10
                },
                {
                    label: "Project Match",
                    data: data.projectMatchSkill,
                    backgroundColor: "rgba(255, 198, 77, 0.75)",
                    borderColor: "#ffc64d",
                    borderWidth: 2,
                    borderRadius: 10
                },
                {
                    label: "ML Suitability",
                    data: data.mlSuitability,
                    backgroundColor: "rgba(255, 138, 31, 0.75)",
                    borderColor: "#ff8a1f",
                    borderWidth: 2,
                    borderRadius: 10
                }
            ];
        }

        if (selectedSkill === "management") {
            chartTitle.textContent = "Management Skill vs Project Match";

            return [
                {
                    label: "Management Skill",
                    data: data.managementSkill,
                    backgroundColor: "rgba(255, 79, 134, 0.75)",
                    borderColor: "#ff4f86",
                    borderWidth: 2,
                    borderRadius: 10
                },
                {
                    label: "Project Match",
                    data: data.projectMatchSkill,
                    backgroundColor: "rgba(255, 198, 77, 0.75)",
                    borderColor: "#ffc64d",
                    borderWidth: 2,
                    borderRadius: 10
                },
                {
                    label: "ML Suitability",
                    data: data.mlSuitability,
                    backgroundColor: "rgba(255, 138, 31, 0.75)",
                    borderColor: "#ff8a1f",
                    borderWidth: 2,
                    borderRadius: 10
                }
            ];
        }

        chartTitle.textContent = "Overall Recommendation Explanation";

        return [
            {
                label: "Project Match",
                data: data.projectMatchSkill,
                backgroundColor: "rgba(255, 198, 77, 0.75)",
                borderColor: "#ffc64d",
                borderWidth: 2,
                borderRadius: 10
            },
            {
                label: "ML Suitability",
                data: data.mlSuitability,
                backgroundColor: "rgba(255, 138, 31, 0.75)",
                borderColor: "#ff8a1f",
                borderWidth: 2,
                borderRadius: 10
            },
            {
                label: "Programming Skill",
                data: data.programmingSkill,
                backgroundColor: "rgba(24, 119, 242, 0.75)",
                borderColor: "#1877f2",
                borderWidth: 2,
                borderRadius: 10
            },
            {
                label: "Management Skill",
                data: data.managementSkill,
                backgroundColor: "rgba(255, 79, 134, 0.75)",
                borderColor: "#ff4f86",
                borderWidth: 2,
                borderRadius: 10
            }
        ];
    }

    function renderBarChart(data) {
        chartInstance = new Chart(canvas, {
            type: "bar",
            data: {
                labels: data.labels,
                datasets: getSelectedDatasets(data)
            },
            options: commonOptions()
        });
    }

    function renderChart() {
        destroyOldChart();
        const data = getChartData();
        updateCards(data);
        renderBarChart(data);
    }

    employeeSelect.addEventListener("change", renderChart);
    skillSelect.addEventListener("change", renderChart);

    renderChart();
});
