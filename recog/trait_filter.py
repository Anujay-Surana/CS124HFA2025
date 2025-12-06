# trait_filter.py
"""
Filtering system for analytics data based on demographic traits.

This module provides filtering capabilities for:
- Age ranges
- Gender
- Ethnicity/Race
- Combined filters

Can be used standalone or integrated with the web API.
"""

import json
import os
from collections import defaultdict


class TraitFilter:
    """
    Filter tracked persons by demographic traits.
    """

    def __init__(self, analytics_dir=None):
        """
        Initialize trait filter.

        Args:
            analytics_dir: Directory containing analytics and person data
        """
        if analytics_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(script_dir, "output")
            self.analytics_dir = os.path.join(self.output_dir, "analytics")
        else:
            self.analytics_dir = analytics_dir
            self.output_dir = os.path.dirname(analytics_dir)

    def load_person_traits(self, person_id):
        """
        Load demographic traits for a person.

        Args:
            person_id: Person ID

        Returns:
            Dictionary with traits or None if not found
        """
        person_dir = os.path.join(self.output_dir, f"person_{person_id:03d}")
        traits_path = os.path.join(person_dir, "traits.json")

        if os.path.exists(traits_path):
            with open(traits_path, "r") as f:
                return json.load(f)

        return None

    def load_all_persons_with_traits(self):
        """
        Load all persons and their traits.

        Returns:
            List of dictionaries with person data and traits
        """
        # Load analytics summary
        summary_path = os.path.join(self.analytics_dir, "analytics_summary.json")
        if not os.path.exists(summary_path):
            return []

        with open(summary_path, "r") as f:
            summary = json.load(f)

        persons = []
        for person in summary.get("persons", []):
            person_id = person["id"]
            traits = self.load_person_traits(person_id)

            person_data = {
                **person,
                "traits": traits if traits else {
                    "age": "Unknown",
                    "gender": "Unknown",
                    "ethnicity": "Unknown"
                }
            }
            persons.append(person_data)

        return persons

    def filter_by_age(self, persons, age_ranges):
        """
        Filter persons by age range.

        Args:
            persons: List of person dictionaries
            age_ranges: List of age ranges to include, e.g., ["(25-32)", "(38-43)"]
                       or "all" for all ages

        Returns:
            Filtered list of persons
        """
        if age_ranges == "all" or not age_ranges:
            return persons

        if isinstance(age_ranges, str):
            age_ranges = [age_ranges]

        filtered = []
        for person in persons:
            person_age = person.get("traits", {}).get("age", "Unknown")
            if person_age in age_ranges or person_age == "Unknown":
                filtered.append(person)

        return filtered

    def filter_by_gender(self, persons, genders):
        """
        Filter persons by gender.

        Args:
            persons: List of person dictionaries
            genders: List of genders to include, e.g., ["Male", "Female"]
                    or "all" for all genders

        Returns:
            Filtered list of persons
        """
        if genders == "all" or not genders:
            return persons

        if isinstance(genders, str):
            genders = [genders]

        filtered = []
        for person in persons:
            person_gender = person.get("traits", {}).get("gender", "Unknown")
            if person_gender in genders or person_gender == "Unknown":
                filtered.append(person)

        return filtered

    def filter_by_ethnicity(self, persons, ethnicities):
        """
        Filter persons by ethnicity/race.

        Args:
            persons: List of person dictionaries
            ethnicities: List of ethnicities to include, e.g., ["Caucasian", "East Asian"]
                        or "all" for all ethnicities

        Returns:
            Filtered list of persons
        """
        if ethnicities == "all" or not ethnicities:
            return persons

        if isinstance(ethnicities, str):
            ethnicities = [ethnicities]

        filtered = []
        for person in persons:
            person_ethnicity = person.get("traits", {}).get("ethnicity", "Unknown")
            if person_ethnicity in ethnicities or person_ethnicity == "Unknown":
                filtered.append(person)

        return filtered

    def filter_by_traits(self, age_ranges=None, genders=None, ethnicities=None):
        """
        Filter persons by multiple demographic traits.

        Args:
            age_ranges: Age ranges to filter by (None = all)
            genders: Genders to filter by (None = all)
            ethnicities: Ethnicities to filter by (None = all)

        Returns:
            Filtered list of persons
        """
        persons = self.load_all_persons_with_traits()

        # Apply filters sequentially
        if age_ranges:
            persons = self.filter_by_age(persons, age_ranges)

        if genders:
            persons = self.filter_by_gender(persons, genders)

        if ethnicities:
            persons = self.filter_by_ethnicity(persons, ethnicities)

        return persons

    def get_demographic_summary(self, persons=None):
        """
        Get demographic breakdown of persons.

        Args:
            persons: List of persons (if None, loads all)

        Returns:
            Dictionary with demographic statistics
        """
        if persons is None:
            persons = self.load_all_persons_with_traits()

        age_counts = defaultdict(int)
        gender_counts = defaultdict(int)
        ethnicity_counts = defaultdict(int)

        for person in persons:
            traits = person.get("traits", {})
            age_counts[traits.get("age", "Unknown")] += 1
            gender_counts[traits.get("gender", "Unknown")] += 1
            ethnicity_counts[traits.get("ethnicity", "Unknown")] += 1

        return {
            "total_persons": len(persons),
            "age_distribution": dict(age_counts),
            "gender_distribution": dict(gender_counts),
            "ethnicity_distribution": dict(ethnicity_counts)
        }

    def get_filtered_analytics(self, age_ranges=None, genders=None, ethnicities=None):
        """
        Get analytics for filtered subset of persons.

        Args:
            age_ranges: Age ranges to include
            genders: Genders to include
            ethnicities: Ethnicities to include

        Returns:
            Dictionary with filtered analytics
        """
        filtered_persons = self.filter_by_traits(age_ranges, genders, ethnicities)

        if not filtered_persons:
            return {
                "filtered_count": 0,
                "total_dwell_time": 0,
                "average_dwell_time": 0,
                "total_visits": 0,
                "persons": []
            }

        total_dwell = sum(p.get("dwell_time_seconds", 0) for p in filtered_persons)
        total_visits = sum(p.get("visit_count", 0) for p in filtered_persons)

        return {
            "filtered_count": len(filtered_persons),
            "total_dwell_time": total_dwell,
            "average_dwell_time": total_dwell / len(filtered_persons),
            "total_visits": total_visits,
            "persons": filtered_persons,
            "demographic_summary": self.get_demographic_summary(filtered_persons)
        }

    def export_filtered_data(self, output_file, age_ranges=None, genders=None, ethnicities=None):
        """
        Export filtered data to JSON file.

        Args:
            output_file: Path to output file
            age_ranges: Age ranges to include
            genders: Genders to include
            ethnicities: Ethnicities to include
        """
        analytics = self.get_filtered_analytics(age_ranges, genders, ethnicities)

        with open(output_file, "w") as f:
            json.dump(analytics, f, indent=2)

        print(f"Filtered data exported to: {output_file}")
        print(f"Filtered persons: {analytics['filtered_count']}")


def save_person_traits(person_dir, age, gender, ethnicity, age_confidence=0.0,
                       gender_confidence=0.0, ethnicity_confidence=0.0):
    """
    Save demographic traits for a person.

    Args:
        person_dir: Person directory path
        age: Age range
        gender: Gender
        ethnicity: Ethnicity
        age_confidence: Age detection confidence (optional)
        gender_confidence: Gender detection confidence (optional)
        ethnicity_confidence: Ethnicity detection confidence (optional)
    """
    traits_path = os.path.join(person_dir, "traits.json")

    traits = {
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "age_confidence": float(age_confidence),
        "gender_confidence": float(gender_confidence),
        "ethnicity_confidence": float(ethnicity_confidence)
    }

    with open(traits_path, "w") as f:
        json.dump(traits, f, indent=2)


# Command-line interface
if __name__ == "__main__":
    filter_system = TraitFilter()

    print("="*60)
    print("Trait-Based Filtering System")
    print("="*60)

    # Load all persons
    all_persons = filter_system.load_all_persons_with_traits()
    print(f"\nTotal persons detected: {len(all_persons)}")

    # Show demographic summary
    print("\nDemographic Summary:")
    summary = filter_system.get_demographic_summary()
    print(f"  Age Distribution: {summary['age_distribution']}")
    print(f"  Gender Distribution: {summary['gender_distribution']}")
    print(f"  Ethnicity Distribution: {summary['ethnicity_distribution']}")

    # Interactive filtering
    print("\n" + "="*60)
    print("Interactive Filtering")
    print("="*60)

    # Age filter
    print("\nAvailable age ranges:")
    age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    for i, age in enumerate(age_ranges, 1):
        print(f"  {i}. {age}")
    print("  Enter comma-separated numbers (or 'all' for all ages): ", end="")
    age_input = input().strip()

    if age_input.lower() == "all" or not age_input:
        selected_ages = None
    else:
        try:
            indices = [int(x.strip()) - 1 for x in age_input.split(",")]
            selected_ages = [age_ranges[i] for i in indices if 0 <= i < len(age_ranges)]
        except:
            print("Invalid input, using all ages")
            selected_ages = None

    # Gender filter
    print("\nGender filter:")
    print("  1. Male")
    print("  2. Female")
    print("  Enter comma-separated numbers (or 'all' for all genders): ", end="")
    gender_input = input().strip()

    if gender_input.lower() == "all" or not gender_input:
        selected_genders = None
    else:
        gender_map = {1: "Male", 2: "Female"}
        try:
            indices = [int(x.strip()) for x in gender_input.split(",")]
            selected_genders = [gender_map[i] for i in indices if i in gender_map]
        except:
            print("Invalid input, using all genders")
            selected_genders = None

    # Ethnicity filter
    print("\nEthnicity filter:")
    ethnicities = ['African', 'Caucasian', 'East Asian', 'South Asian', 'Hispanic/Latino', 'Middle Eastern']
    for i, eth in enumerate(ethnicities, 1):
        print(f"  {i}. {eth}")
    print("  Enter comma-separated numbers (or 'all' for all): ", end="")
    ethnicity_input = input().strip()

    if ethnicity_input.lower() == "all" or not ethnicity_input:
        selected_ethnicities = None
    else:
        try:
            indices = [int(x.strip()) - 1 for x in ethnicity_input.split(",")]
            selected_ethnicities = [ethnicities[i] for i in indices if 0 <= i < len(ethnicities)]
        except:
            print("Invalid input, using all ethnicities")
            selected_ethnicities = None

    # Apply filters
    print("\n" + "="*60)
    print("Filtering Results")
    print("="*60)

    results = filter_system.get_filtered_analytics(
        age_ranges=selected_ages,
        genders=selected_genders,
        ethnicities=selected_ethnicities
    )

    print(f"\nFiltered persons: {results['filtered_count']}")
    print(f"Total dwell time: {results['total_dwell_time']:.1f}s")
    print(f"Average dwell time: {results['average_dwell_time']:.1f}s")
    print(f"Total visits: {results['total_visits']}")

    print("\nFiltered demographic breakdown:")
    demo = results['demographic_summary']
    print(f"  Age: {demo['age_distribution']}")
    print(f"  Gender: {demo['gender_distribution']}")
    print(f"  Ethnicity: {demo['ethnicity_distribution']}")

    # Show person details
    if results['persons']:
        print("\nPerson Details:")
        for person in results['persons'][:5]:  # Show first 5
            traits = person.get('traits', {})
            print(f"\n  Person {person['id']}:")
            print(f"    Age: {traits.get('age', 'Unknown')}")
            print(f"    Gender: {traits.get('gender', 'Unknown')}")
            print(f"    Ethnicity: {traits.get('ethnicity', 'Unknown')}")
            print(f"    Dwell time: {person.get('dwell_time_seconds', 0):.1f}s")
            print(f"    Visits: {person.get('visit_count', 0)}")

        if len(results['persons']) > 5:
            print(f"\n  ... and {len(results['persons']) - 5} more persons")

    # Export option
    print("\n" + "="*60)
    print("Export filtered data? (y/n): ", end="")
    export_input = input().strip().lower()

    if export_input == 'y':
        output_file = os.path.join(filter_system.analytics_dir, "filtered_analytics.json")
        filter_system.export_filtered_data(
            output_file,
            age_ranges=selected_ages,
            genders=selected_genders,
            ethnicities=selected_ethnicities
        )
