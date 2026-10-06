"""Shared feature definitions."""
FEATURES = [
    "age", "gpa", "attendance_rate", "failed_courses", "credits_passed_ratio",
    "assignments_submitted_rate", "lms_logins_per_week", "scholarship",
    "tuition_paid_on_time", "family_income_level", "works_part_time",
    "first_generation", "commute_km",
]

# Human-readable names for explanations
LABELS = {
    "age": "Age",
    "gpa": "GPA",
    "attendance_rate": "Attendance rate",
    "failed_courses": "Failed courses",
    "credits_passed_ratio": "Share of credits passed",
    "assignments_submitted_rate": "Assignment submission rate",
    "lms_logins_per_week": "Online platform logins/week",
    "scholarship": "Has scholarship",
    "tuition_paid_on_time": "Tuition paid on time",
    "family_income_level": "Family income level",
    "works_part_time": "Works part-time",
    "first_generation": "First-generation student",
    "commute_km": "Commute distance (km)",
}
