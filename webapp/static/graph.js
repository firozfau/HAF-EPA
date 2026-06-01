const recommendations = window.recommendations || [];

document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("recommendationChart");
    const employeeSelect = document.getElementById("employeeSelect");
    const skillSelect = document.getElementById("skillSelect");
    const chartTypeSelect = document.getElementById("chartType");
    const chartTitle = document.getElementById("chartTitle");

    if (!canvas || !employeeSelect || !skillSelect || !chartTypeSelect || !chartTitle) {
        console.error("Graph required HTML elements are missing.");
        return;
    }

    let chartInstance = null;

    const chartColors = {
        programmingSkill: "#1877f2",
        managementSkill: "#ff4f86",
        projectMatchSkill: "#ffc64d"
    };

    function getPercentageValue(...values) {
        for (const value of values) {
            const numericValue = Number(value);

            if (!Number.isNaN(numericValue) && numericValue > 0) {
                return numericValue;
            }
        }

        return 0;
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
            labels: filteredData.map(item => item.full_name),

            // Management Skill should show the employee profile management score.
            // Do not use only management_project_match_percentage here, because that value
            // becomes 0 when the uploaded PDF has no management-category required skill.
            managementSkill: filteredData.map(item =>
                getPercentageValue(item.management_score, item.management_skill_percentage, item.management_project_match_percentage)
            ),

            // Programming Skill should show the employee profile programming score.
            // PDF-based programming_project_match_percentage is used only as fallback.
            programmingSkill: filteredData.map(item =>
                getPercentageValue(item.programming_score, item.programming_skill_percentage, item.programming_project_match_percentage)
            ),

            // Project Match Skill is the actual required-skill match from uploaded PDF.
            projectMatchSkill: filteredData.map(item =>
                getPercentageValue(item.overall_project_match_percentage, item.match_percentage)
            )
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

        const programmingAverage = data.programmingSkill.length
            ? data.programmingSkill.reduce((a, b) => a + b, 0) / data.programmingSkill.length
            : 0;

        const managementAverage = data.managementSkill.length
            ? data.managementSkill.reduce((a, b) => a + b, 0) / data.managementSkill.length
            : 0;

        document.getElementById("totalEmployeesCard").textContent = total;
        document.getElementById("topMatchCard").textContent = `${topMatch.toFixed(2)}%`;
        document.getElementById("programmingSkillCard").textContent = `${programmingAverage.toFixed(2)}%`;
        document.getElementById("managementSkillCard").textContent = `${managementAverage.toFixed(2)}%`;
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
            chartTitle.textContent = "Programming Skill Analysis";

            return [
                {
                    label: "Programming Skill",
                    data: data.programmingSkill,
                    backgroundColor: "rgba(24, 119, 242, 0.75)",
                    borderColor: chartColors.programmingSkill,
                    borderWidth: 2,
                    borderRadius: 10
                },
                {
                    label: "Project Match Skill",
                    data: data.projectMatchSkill,
                    backgroundColor: "rgba(255, 198, 77, 0.75)",
                    borderColor: chartColors.projectMatchSkill,
                    borderWidth: 2,
                    borderRadius: 10
                }
            ];
        }

        if (selectedSkill === "management") {
            chartTitle.textContent = "Management Skill Analysis";

            return [
                {
                    label: "Management Skill",
                    data: data.managementSkill,
                    backgroundColor: "rgba(255, 79, 134, 0.75)",
                    borderColor: chartColors.managementSkill,
                    borderWidth: 2,
                    borderRadius: 10
                },
                {
                    label: "Project Match Skill",
                    data: data.projectMatchSkill,
                    backgroundColor: "rgba(255, 198, 77, 0.75)",
                    borderColor: chartColors.projectMatchSkill,
                    borderWidth: 2,
                    borderRadius: 10
                }
            ];
        }

        chartTitle.textContent = "Programming vs Management Skill Comparison";

        return [
            {
                label: "Management Skill",
                data: data.managementSkill,
                backgroundColor: "rgba(255, 79, 134, 0.75)",
                borderColor: chartColors.managementSkill,
                borderWidth: 2,
                borderRadius: 10
            },
            {
                label: "Programming Skill",
                data: data.programmingSkill,
                backgroundColor: "rgba(24, 119, 242, 0.75)",
                borderColor: chartColors.programmingSkill,
                borderWidth: 2,
                borderRadius: 10
            },
            {
                label: "Project Match Skill",
                data: data.projectMatchSkill,
                backgroundColor: "rgba(255, 198, 77, 0.75)",
                borderColor: chartColors.projectMatchSkill,
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

    function renderLineChart(data) {
        const lineDatasets = getSelectedDatasets(data).map(dataset => ({
            label: dataset.label,
            data: dataset.data,
            borderColor: dataset.borderColor,
            backgroundColor: dataset.backgroundColor,
            pointBackgroundColor: dataset.borderColor,
            borderWidth: 3,
            pointRadius: 5,
            tension: 0.35,
            fill: false
        }));

        chartInstance = new Chart(canvas, {
            type: "line",
            data: {
                labels: data.labels,
                datasets: lineDatasets
            },
            options: commonOptions()
        });
    }

    function renderPieChart(data) {
        chartTitle.textContent = "Project Match Distribution";

        chartInstance = new Chart(canvas, {
            type: "doughnut",
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: "Project Match Skill",
                        data: data.projectMatchSkill,
                        backgroundColor: [
                            "#1877f2",
                            "#ff4f86",
                            "#ff8a1f",
                            "#ffc64d",
                            "#16a34a",
                            "#7c3aed",
                            "#06b6d4",
                            "#ef4444",
                            "#84cc16",
                            "#f97316"
                        ],
                        borderColor: "#ffffff",
                        borderWidth: 4,
                        hoverOffset: 12
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "58%"
            }
        });
    }

    function renderChart() {
        destroyOldChart();

        const data = getChartData();

        updateCards(data);

        if (chartTypeSelect.value === "bar") {
            renderBarChart(data);
        }

        if (chartTypeSelect.value === "line") {
            renderLineChart(data);
        }

        if (chartTypeSelect.value === "pie") {
            renderPieChart(data);
        }
    }

    employeeSelect.addEventListener("change", renderChart);
    skillSelect.addEventListener("change", renderChart);
    chartTypeSelect.addEventListener("change", renderChart);

    renderChart();
});