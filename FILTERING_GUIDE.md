# Demographic Trait Filtering Guide

This guide explains how to filter analytics data by age, gender, and ethnicity.

## Overview

The filtering system allows you to:
- Filter tracked persons by age range, gender, and/or ethnicity
- Get analytics for specific demographic groups
- Export filtered data
- Use filters via command-line or web API

## Quick Start

### Option 1: Interactive Command-Line

```bash
python3 recog/trait_filter.py
```

This starts an interactive session where you can:
1. Select age ranges to include
2. Select genders to include
3. Select ethnicities to include
4. View filtered results
5. Export filtered data

### Option 2: Web API

Start the API server:
```bash
python3 recog/web_api.py
```

Then use the filtering endpoints (see API section below).

## Demographic Categories

### Age Ranges
- `(0-2)` - Infant
- `(4-6)` - Toddler
- `(8-12)` - Child
- `(15-20)` - Teen
- `(25-32)` - Young Adult
- `(38-43)` - Adult
- `(48-53)` - Middle Age
- `(60-100)` - Senior

### Gender
- `Male`
- `Female`

### Ethnicity/Race
- `African`
- `Caucasian`
- `East Asian`
- `South Asian`
- `Hispanic/Latino`
- `Middle Eastern`

## Data Format

Trait data is stored per person in `output/person_XXX/traits.json`:

```json
{
  "age": "(25-32)",
  "gender": "Male",
  "ethnicity": "Caucasian",
  "age_confidence": 0.85,
  "gender_confidence": 0.92,
  "ethnicity_confidence": 0.78
}
```

## Python API

### Basic Filtering

```python
from trait_filter import TraitFilter

# Initialize filter
filter_system = TraitFilter()

# Filter by age only
young_adults = filter_system.filter_by_traits(
    age_ranges=["(25-32)", "(38-43)"]
)

# Filter by gender only
males = filter_system.filter_by_traits(
    genders=["Male"]
)

# Filter by ethnicity only
asian_persons = filter_system.filter_by_traits(
    ethnicities=["East Asian", "South Asian"]
)

# Combine multiple filters
filtered = filter_system.filter_by_traits(
    age_ranges=["(25-32)"],
    genders=["Female"],
    ethnicities=["Caucasian"]
)
```

### Get Analytics for Filtered Group

```python
# Get complete analytics for filtered subset
results = filter_system.get_filtered_analytics(
    age_ranges=["(25-32)", "(38-43)"],
    genders=["Male"]
)

print(f"Filtered count: {results['filtered_count']}")
print(f"Average dwell time: {results['average_dwell_time']:.1f}s")
print(f"Total visits: {results['total_visits']}")

# Access demographic breakdown
demo = results['demographic_summary']
print(f"Age distribution: {demo['age_distribution']}")
print(f"Gender distribution: {demo['gender_distribution']}")
```

### Get Demographic Summary

```python
# Get overall demographic breakdown
summary = filter_system.get_demographic_summary()

print(f"Total persons: {summary['total_persons']}")
print(f"Age distribution: {summary['age_distribution']}")
print(f"Gender distribution: {summary['gender_distribution']}")
print(f"Ethnicity distribution: {summary['ethnicity_distribution']}")
```

### Export Filtered Data

```python
# Export filtered data to JSON
filter_system.export_filtered_data(
    "filtered_output.json",
    age_ranges=["(25-32)"],
    genders=["Male"]
)
```

### Save Person Traits

When integrating with detection systems:

```python
from trait_filter import save_person_traits

# Save traits for a person
save_person_traits(
    person_dir="output/person_000",
    age="(25-32)",
    gender="Male",
    ethnicity="Caucasian",
    age_confidence=0.85,
    gender_confidence=0.92,
    ethnicity_confidence=0.78
)
```

## Web API Endpoints

### Filter Persons

**Endpoint:** `GET /api/filter/persons`

**Query Parameters:**
- `age` - Comma-separated age ranges
- `gender` - Comma-separated genders
- `ethnicity` - Comma-separated ethnicities

**Examples:**

```bash
# Filter by age
curl "http://localhost:5000/api/filter/persons?age=(25-32),(38-43)"

# Filter by gender
curl "http://localhost:5000/api/filter/persons?gender=Male"

# Filter by ethnicity
curl "http://localhost:5000/api/filter/persons?ethnicity=Caucasian,East%20Asian"

# Combine filters
curl "http://localhost:5000/api/filter/persons?age=(25-32)&gender=Female&ethnicity=Caucasian"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "filtered_count": 5,
    "total_dwell_time": 1523.5,
    "average_dwell_time": 304.7,
    "total_visits": 12,
    "persons": [
      {
        "id": 0,
        "dwell_time_seconds": 307.5,
        "visit_count": 2,
        "traits": {
          "age": "(25-32)",
          "gender": "Female",
          "ethnicity": "Caucasian"
        }
      }
    ],
    "demographic_summary": {
      "total_persons": 5,
      "age_distribution": {"(25-32)": 5},
      "gender_distribution": {"Female": 5},
      "ethnicity_distribution": {"Caucasian": 5}
    }
  }
}
```

### Get Demographic Summary

**Endpoint:** `GET /api/demographics/summary`

**Example:**

```bash
curl "http://localhost:5000/api/demographics/summary"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "total_persons": 15,
    "age_distribution": {
      "(25-32)": 7,
      "(38-43)": 5,
      "(48-53)": 3
    },
    "gender_distribution": {
      "Male": 8,
      "Female": 7
    },
    "ethnicity_distribution": {
      "Caucasian": 6,
      "East Asian": 4,
      "Hispanic/Latino": 3,
      "African": 2
    }
  }
}
```

## Use Cases

### Retail Analytics

**Example:** Track shopping patterns by age group

```python
# Get analytics for young adults
young_shoppers = filter_system.get_filtered_analytics(
    age_ranges=["(15-20)", "(25-32)"]
)

print(f"Young shoppers: {young_shoppers['filtered_count']}")
print(f"Average browse time: {young_shoppers['average_dwell_time']:.1f}s")
```

### Marketing Analysis

**Example:** Compare male vs female engagement

```python
# Get male analytics
male_analytics = filter_system.get_filtered_analytics(genders=["Male"])

# Get female analytics
female_analytics = filter_system.get_filtered_analytics(genders=["Female"])

print(f"Male avg dwell: {male_analytics['average_dwell_time']:.1f}s")
print(f"Female avg dwell: {female_analytics['average_dwell_time']:.1f}s")
```

### Demographic Targeting

**Example:** Find most engaged demographic

```python
from trait_filter import TraitFilter

filter_system = TraitFilter()

# Try all age ranges
age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(25-32)', '(38-43)', '(48-53)', '(60-100)']

best_age = None
best_dwell = 0

for age in age_ranges:
    results = filter_system.get_filtered_analytics(age_ranges=[age])
    if results['filtered_count'] > 0:
        avg_dwell = results['average_dwell_time']
        if avg_dwell > best_dwell:
            best_dwell = avg_dwell
            best_age = age

print(f"Most engaged age group: {best_age}")
print(f"Average dwell time: {best_dwell:.1f}s")
```

## Integration with Detection System

To integrate demographic detection with analytics:

### Step 1: Detect Demographics

Use your detection system (e.g., DetectionMain.py) to get age, gender, ethnicity.

### Step 2: Save Traits

When saving person data in your tracking system:

```python
from trait_filter import save_person_traits

# After detecting a person
person_dir = ensure_person_dir(OUTPUT_DIR, person_id)

# Save demographic traits
save_person_traits(
    person_dir,
    age=detected_age,
    gender=detected_gender,
    ethnicity=detected_ethnicity,
    age_confidence=age_conf,
    gender_confidence=gender_conf,
    ethnicity_confidence=eth_conf
)
```

### Step 3: Use Filters

Now you can filter using the trait system:

```python
# Filter by detected traits
filtered = filter_system.filter_by_traits(
    age_ranges=["(25-32)"],
    genders=["Male"]
)
```

## Command-Line Examples

### Interactive Mode

```bash
$ python3 recog/trait_filter.py

============================================================
Trait-Based Filtering System
============================================================

Total persons detected: 15

Demographic Summary:
  Age Distribution: {'(25-32)': 7, '(38-43)': 5, '(48-53)': 3}
  Gender Distribution: {'Male': 8, 'Female': 7}
  Ethnicity Distribution: {'Caucasian': 6, 'East Asian': 4, ...}

============================================================
Interactive Filtering
============================================================

Available age ranges:
  1. (0-2)
  2. (4-6)
  3. (8-12)
  4. (15-20)
  5. (25-32)
  6. (38-43)
  7. (48-53)
  8. (60-100)
  Enter comma-separated numbers (or 'all' for all ages): 5,6

Gender filter:
  1. Male
  2. Female
  Enter comma-separated numbers (or 'all' for all genders): 1

Ethnicity filter:
  1. African
  2. Caucasian
  3. East Asian
  4. South Asian
  5. Hispanic/Latino
  6. Middle Eastern
  Enter comma-separated numbers (or 'all' for all): all

============================================================
Filtering Results
============================================================

Filtered persons: 8
Total dwell time: 2450.3s
Average dwell time: 306.3s
Total visits: 20

Export filtered data? (y/n): y
Filtered data exported to: recog/output/analytics/filtered_analytics.json
```

## JavaScript/Web Dashboard Integration

### Fetch Filtered Data

```javascript
// Filter by age and gender
async function fetchFilteredData() {
    const params = new URLSearchParams({
        age: '(25-32),(38-43)',
        gender: 'Male'
    });

    const response = await fetch(`http://localhost:5000/api/filter/persons?${params}`);
    const data = await response.json();

    console.log(`Filtered count: ${data.data.filtered_count}`);
    console.log(`Avg dwell: ${data.data.average_dwell_time}s`);
}

// Get demographic breakdown
async function getDemographics() {
    const response = await fetch('http://localhost:5000/api/demographics/summary');
    const data = await response.json();

    const demo = data.data;
    console.log('Age distribution:', demo.age_distribution);
    console.log('Gender distribution:', demo.gender_distribution);
}
```

### Dynamic Filter UI

```html
<select id="ageFilter" multiple>
    <option value="(25-32)">25-32</option>
    <option value="(38-43)">38-43</option>
    <!-- ... -->
</select>

<select id="genderFilter" multiple>
    <option value="Male">Male</option>
    <option value="Female">Female</option>
</select>

<button onclick="applyFilters()">Filter</button>

<script>
async function applyFilters() {
    const ages = Array.from(document.getElementById('ageFilter').selectedOptions)
        .map(opt => opt.value).join(',');
    const genders = Array.from(document.getElementById('genderFilter').selectedOptions)
        .map(opt => opt.value).join(',');

    const params = new URLSearchParams();
    if (ages) params.append('age', ages);
    if (genders) params.append('gender', genders);

    const response = await fetch(`http://localhost:5000/api/filter/persons?${params}`);
    const data = await response.json();

    // Display filtered results
    displayResults(data.data);
}
</script>
```

## Notes

- **Trait data is optional** - Filtering works even if some persons don't have trait data (they're included as "Unknown")
- **Confidence scores** - Each trait includes a confidence score from the detection model
- **Multiple filters** - Filters are combined with AND logic (person must match ALL specified criteria)
- **Empty results** - If no persons match the filters, you'll get empty results with count=0

## Troubleshooting

**No trait data found:**
- Traits must be saved using `save_person_traits()` function
- Check that `traits.json` exists in person directories
- Integrate trait saving into your detection pipeline

**Filter returns everyone:**
- Make sure you're passing filter parameters correctly
- Check that trait data is being saved properly
- Verify traits.json files contain correct data

**API returns 500 error:**
- Check that analytics_summary.json exists
- Verify person directories are formatted correctly (person_XXX)
- Check Flask console for detailed error messages

## Complete Example

```python
from trait_filter import TraitFilter, save_person_traits
import os

# Setup
filter_system = TraitFilter()

# Simulate saving traits for detected persons
for person_id in range(5):
    person_dir = f"recog/output/person_{person_id:03d}"
    os.makedirs(person_dir, exist_ok=True)

    # Save example traits
    save_person_traits(
        person_dir,
        age="(25-32)" if person_id % 2 == 0 else "(38-43)",
        gender="Male" if person_id < 3 else "Female",
        ethnicity="Caucasian",
        age_confidence=0.85,
        gender_confidence=0.90,
        ethnicity_confidence=0.75
    )

# Now filter
young_males = filter_system.filter_by_traits(
    age_ranges=["(25-32)"],
    genders=["Male"]
)

print(f"Found {len(young_males)} young male persons")

# Get analytics
analytics = filter_system.get_filtered_analytics(
    age_ranges=["(25-32)"],
    genders=["Male"]
)

print(f"Analytics: {analytics}")
```

This filtering system gives you powerful demographic analysis capabilities for your face recognition analytics!
