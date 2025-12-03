#!/usr/bin/env python3
"""
Analyze which YOLO model best detects furniture and home objects from a given list.
"""

from ultralytics import YOLO
import os

# Your list of interest (furniture and home items)
INTEREST_CLASSES = [
    "Sofa", "Couch", "Sectional", "Loveseat", "Chaise Lounge", "Recliner", "Futon", "Daybed",
    "Sleeper Sofa", "Sofa Bed", "Modular Sofa", "Pull-out Sofa", "Fold-out Couch", "Corner Sofa",
    "Lounge Sofa", "Tufted Sofa", "Leather Sofa", "Fabric Sofa", "Retro Sofa", "Contemporary Sofa",
    "Armchair", "Accent Chair", "Dining Chair", "Office Chair", "Gaming Chair", "Task Chair",
    "Lounge Chair", "Rocking Chair", "Wingback Chair", "Swivel Chair", "Folding Chair",
    "Reclining Chair", "Barstool", "Counter Stool", "Drafting Chair", "Club Chair",
    "Wicker Chair", "Rattan Chair", "Bamboo Chair", "Upholstered Chair", "Stackable Chair",
    "Patio Chair", "Adirondack Chair", "Sling Chair", "Beanbag Chair", "Hammock Chair",
    "Hanging Chair", "Sun Lounger", "Folding Armchair", "Padded Chair", "Stool",
    "Step Stool", "Folding Stool", "Vanity Stool", "Bar Stool", "Pouf", "Ottoman",
    "Footstool", "Tuffet", "Twin Bed", "Full Bed", "Queen Bed", "King Bed", "Bunk Bed",
    "Trundle Bed", "Loft Bed", "Canopy Bed", "Four-poster Bed", "Platform Bed", "Murphy Bed",
    "Floating Bed", "Storage Bed", "Bed Frame", "Headboard", "Footboard", "Mattress",
    "Box Spring", "Mattress Topper", "Mattress Protector", "Bed Skirt", "Comforter",
    "Blanket", "Quilt", "Duvet", "Pillow", "Throw Pillow", "Body Pillow", "Sheet Set",
    "Pillowcase", "Bedspread", "Nightstand", "Bedside Table", "Dresser", "Chest of Drawers",
    "Tallboy", "Vanity", "Dressing Table", "Makeup Table", "Changing Table", "Wardrobe",
    "Armoire", "Closet Organizer", "Freestanding Closet", "Open Wardrobe", "Storage Cabinet",
    "Cabinet", "Drawer Unit", "Sideboard", "Buffet", "Credenza", "Cupboard", "Storage Chest",
    "Blanket Chest", "Toy Chest", "Trunk", "Shoe Cabinet", "Shoe Rack", "Filing Cabinet",
    "Rolling Cabinet", "Pantry Cabinet", "Bookshelf", "Bookcase", "Wall Shelf", "Corner Shelf",
    "Floating Shelf", "Ladder Shelf", "Cube Shelf", "Modular Shelf", "Hanging Shelf",
    "Standing Shelf", "Display Shelf", "Built-in Shelf", "Dining Table", "Kitchen Table",
    "Coffee Table", "End Table", "Side Table", "Console Table", "Entryway Table", "Accent Table",
    "Folding Table", "Bar Table", "Pub Table", "Bistro Table", "Tray Table", "Trolley Table",
    "Writing Desk", "Computer Desk", "Executive Desk", "Corner Desk", "Standing Desk",
    "Adjustable Desk", "Sit-Stand Desk", "Wall-Mounted Desk", "Folding Desk", "Rolltop Desk",
    "Art Desk", "Drafting Table", "Lap Desk", "Bench", "Storage Bench", "Dining Bench",
    "Hall Bench", "Entryway Bench", "Garden Bench", "Outdoor Bench", "Workbench",
    "TV Stand", "Media Console", "Entertainment Center", "TV Cabinet", "Lowboard",
    "Wall Unit", "AV Cabinet", "Patio Sofa", "Porch Swing", "Outdoor Bench", "Lounge Chair",
    "Hammock", "Picnic Table", "Garden Chair", "Folding Lounge", "Outdoor Dining Set",
    "Patio Table", "Poolside Lounger", "Wall Mirror", "Full-Length Mirror", "Vanity Mirror",
    "Decorative Mirror", "Floor Mirror", "Mirror Cabinet", "Laundry Basket", "Laundry Hamper",
    "Hall Tree", "Coat Rack", "Umbrella Stand", "Shoe Bench", "Entry Shelf", "Entry Table",
    "Office Desk", "Office Chair", "File Cabinet", "Desk Organizer", "Monitor Stand",
    "Keyboard Tray", "Printer Stand", "Cubicle Panel", "Office Partition", "Mobile Pedestal",
    "Desk Hutch", "Workstation", "Pen Holder", "Mail Organizer", "Bookend", "Drawer Divider",
    "Under-bed Storage", "Closet Shelf", "Rolling Cart", "Hanging Organizer", "Storage Box",
    "Folding Chair", "Convertible Sofa", "Foldable Table", "Extendable Table", "Nesting Table",
    "Stacking Stool", "Collapsible Shelf", "Lift-top Coffee Table", "Drop-leaf Table",
    "Wall Bed", "Side Console", "Hall Table", "Console Cabinet", "Glass Display Cabinet",
    "China Cabinet", "Curio Cabinet", "Wine Rack", "Bar Cart", "Liquor Cabinet",
    "Cocktail Table", "Toy Box", "Rolling Shelf", "Modular Cabinet", "Cubby Storage",
    "Media Shelf", "Record Cabinet", "Kitchen Island", "Bar Unit", "Plant Stand", "Room Divider"
]

# Normalize class names for matching (lowercase, remove special chars)
def normalize_class(name):
    """Normalize class name for comparison"""
    return name.lower().strip().replace('-', ' ').replace('_', ' ')

# Normalize your interest list
normalized_interest = {normalize_class(c): c for c in INTEREST_CLASSES}

MODELS_DIR = "models"
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]

print("=" * 80)
print("MODEL ANALYSIS: Furniture & Home Objects Detection")
print("=" * 80)
print(f"\nTotal items of interest: {len(INTEREST_CLASSES)}")
print(f"Models to analyze: {len(model_files)}\n")

results = {}

for model_file in sorted(model_files):
    model_path = os.path.join(MODELS_DIR, model_file)
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_file}")
    print(f"{'='*80}")
    
    try:
        model = YOLO(model_path)
        class_names = model.names
        
        # Get all class names from the model
        model_classes = list(class_names.values())
        normalized_model_classes = {normalize_class(c): c for c in model_classes}
        
        print(f"Total classes in model: {len(model_classes)}")
        print(f"Sample classes: {model_classes[:10]}...")
        
        # Find matches
        matches = []
        for norm_interest, original_interest in normalized_interest.items():
            # Direct match
            if norm_interest in normalized_model_classes:
                matches.append((original_interest, normalized_model_classes[norm_interest]))
            else:
                # Partial match (check if any model class contains the interest word or vice versa)
                for norm_model, orig_model in normalized_model_classes.items():
                    # Split into words and check for overlap
                    interest_words = set(norm_interest.split())
                    model_words = set(norm_model.split())
                    
                    # Check for significant word overlap
                    if interest_words and model_words:
                        overlap = interest_words.intersection(model_words)
                        if overlap and len(overlap) >= min(len(interest_words), len(model_words)) * 0.5:
                            matches.append((original_interest, orig_model))
                            break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)
        
        match_count = len(unique_matches)
        match_percentage = (match_count / len(INTEREST_CLASSES)) * 100
        
        results[model_file] = {
            'total_classes': len(model_classes),
            'matches': unique_matches,
            'match_count': match_count,
            'match_percentage': match_percentage,
            'class_names': model_classes
        }
        
        print(f"\n✅ Matches found: {match_count}/{len(INTEREST_CLASSES)} ({match_percentage:.1f}%)")
        print(f"\nTop matches:")
        for i, (interest, model_class) in enumerate(unique_matches[:20], 1):
            print(f"  {i:2d}. '{interest}' → '{model_class}'")
        if len(unique_matches) > 20:
            print(f"  ... and {len(unique_matches) - 20} more")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        results[model_file] = {'error': str(e)}

# Summary
print("\n\n" + "=" * 80)
print("SUMMARY: Best Model for Furniture & Home Objects")
print("=" * 80)
print(f"\n{'Model':<40} {'Total Classes':<15} {'Matches':<10} {'Coverage':<10}")
print("-" * 80)

sorted_results = sorted(
    [(k, v) for k, v in results.items() if 'error' not in v],
    key=lambda x: x[1]['match_count'],
    reverse=True
)

for model_file, data in sorted_results:
    print(f"{model_file:<40} {data['total_classes']:<15} {data['match_count']:<10} {data['match_percentage']:>6.1f}%")

if sorted_results:
    best_model = sorted_results[0][0]
    best_data = sorted_results[0][1]
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   - Matches {best_data['match_count']} out of {len(INTEREST_CLASSES)} items ({best_data['match_percentage']:.1f}%)")
    print(f"   - Total model classes: {best_data['total_classes']}")

