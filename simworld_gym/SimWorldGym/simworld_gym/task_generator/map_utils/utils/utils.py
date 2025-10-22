import matplotlib.pyplot as plt
from simworld_gym import Map, Node, Vector, Config
from typing import List, Optional


def estimated_delivery_time(store_position: Vector, customer_position: Vector):
    distance = (store_position - customer_position).length()
    return (distance / Config.DELIVERY_MAN_MIN_SPEED) * 3

def visualize_map(map_obj: Map, save_path: Optional[str] = None):
    plt.figure(figsize=(12, 8))
    
    for edge in map_obj.edges:
        x_coords = [edge.node1.position.x, edge.node2.position.x]
        y_coords = [-edge.node1.position.y, -edge.node2.position.y]
        plt.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1)
    
    normal_nodes = []
    intersection_nodes = []
    supply_nodes = []
    
    for node in map_obj.nodes:
        if node.type == "normal":
            normal_nodes.append(node)
        elif node.type == "intersection":
            intersection_nodes.append(node)
        elif node.type == "supply":
            supply_nodes.append(node)
    
    if normal_nodes:
        x_coords = [node.position.x for node in normal_nodes]
        y_coords = [-node.position.y for node in normal_nodes]
        plt.scatter(x_coords, y_coords, c='blue', s=30, label='Normal Nodes')
    
    if intersection_nodes:
        x_coords = [node.position.x for node in intersection_nodes]
        y_coords = [-node.position.y for node in intersection_nodes]
        plt.scatter(x_coords, y_coords, c='red', s=50, label='Intersections')
    
    if supply_nodes:
        x_coords = [node.position.x for node in supply_nodes]
        y_coords = [-node.position.y for node in supply_nodes]
        plt.scatter(x_coords, y_coords, c='green', s=80, marker='*', label='Supply Points')
    
    plt.title('Map Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_map_with_path(map_obj: Map, path: List[Node], nodes_of_interest: List[Node], roads: List[dict] = None, save_path: Optional[str] = None):
    plt.figure(figsize=(12, 8))

    for edge in map_obj.edges:
        x_coords = [edge.node1.position.x, edge.node2.position.x]
        y_coords = [-edge.node1.position.y, -edge.node2.position.y] # Invert Y
        plt.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1)

    # --- Plot Road Names from the 'roads' list ---
    plotted_road_names = set() # Keep track of plotted names/locations
    if roads:
        for road in roads:
            try:
                start_pos = road["start"]
                end_pos = road["end"]
                road_name = road["name"]

                # Ensure start/end pos have x, y attributes (adjust if needed)
                if (hasattr(start_pos, 'x') and hasattr(start_pos, 'y') and
                    hasattr(end_pos, 'x') and hasattr(end_pos, 'y')):

                    x_coords = [start_pos.x, end_pos.x]
                    y_coords = [-start_pos.y, -end_pos.y] # Invert Y

                    # Calculate midpoint for text placement
                    mid_x = (x_coords[0] + x_coords[1]) / 2
                    mid_y = (y_coords[0] + y_coords[1]) / 2

                    # Create a key to prevent duplicate plotting in the same spot
                    position_tuple = (round(mid_x, 1), round(mid_y, 1))
                    name_pos_key = (road_name, position_tuple)

                    if name_pos_key not in plotted_road_names:
                        # Determine if the road is vertical
                        is_vertical = abs(x_coords[0] - x_coords[1]) < 1e-5
                        
                        rotation = 90 if is_vertical else 0
                        vertical_alignment = 'center' if is_vertical else 'bottom'
                        
                        plt.text(mid_x, mid_y, road_name,
                                 fontsize=7, color='darkgreen',
                                 ha='center', va=vertical_alignment, # Adjust VA for vertical roads
                                 rotation=rotation) # Apply rotation, Removed bbox
                        plotted_road_names.add(name_pos_key)
                else:
                     print(f"Warning: Skipping road '{road_name}' due to missing coordinate attributes.")

            except (KeyError, AttributeError) as e:
                print(f"Warning: Skipping a road due to missing data or incorrect format: {e}")
    # --- End Plot Road Names ---


    # --- Plot Nodes (Intersections, Supply Points) ---
    normal_nodes = []
    intersection_nodes = []
    supply_nodes = []

    for node in map_obj.nodes:
        if node.type == "normal":
            normal_nodes.append(node)
        elif node.type == "intersection":
            intersection_nodes.append(node)
        elif node.type == "supply":
            supply_nodes.append(node)

    # if normal_nodes:
    #     x_coords = [node.position.x for node in normal_nodes]
    #     y_coords = [node.position.y for node in normal_nodes]
    #     plt.scatter(x_coords, y_coords, c='blue', s=30, label='Normal Nodes')

    if intersection_nodes:
        x_coords = [node.position.x for node in intersection_nodes]
        y_coords = [-node.position.y for node in intersection_nodes]
        plt.scatter(x_coords, y_coords, c='red', s=50, label='Intersections')

    if supply_nodes:
        x_coords = [node.position.x for node in supply_nodes]
        y_coords = [-node.position.y for node in supply_nodes]
        plt.scatter(x_coords, y_coords, c='green', s=80, marker='*', label='Supply Points')

    if path:
        x_path = [node.position.x for node in path]
        y_path = [-node.position.y for node in path]
        plt.plot(x_path, y_path, color='orange', linewidth=2.5, label='Path')
        for i in range(len(x_path)-1):
            plt.annotate('', xy=(x_path[i+1], y_path[i+1]), xytext=(x_path[i], y_path[i]),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2.5))

        plt.scatter(x_path[0], y_path[0], c='black', s=300, marker='o')
        plt.text(x_path[0], y_path[0], 'S', fontsize=15, color='white',
                 ha='center', va='center', weight='bold')

        plt.scatter(x_path[-1], y_path[-1], c='purple', s=300, marker='o')
        plt.text(x_path[-1], y_path[-1], 'E', fontsize=15, color='white',
                 ha='center', va='center', weight='bold')
    
    for i, node in enumerate(nodes_of_interest):
        x_coords = node.x
        y_coords = -node.y
        plt.scatter(x_coords, y_coords, c='magenta', s=300, marker='o', label='Landmark')
        plt.text(x_coords, y_coords, str(i+1), fontsize=15, color='white',
                 ha='center', va='center', weight='bold')

    plt.title('Map with Path Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_map_with_multiagents(map_obj: Map, spawning_locations: List[Node], destination: Node, roads: List[dict] = None, save_path: Optional[str] = None):
    plt.figure(figsize=(12, 8))

    # --- Plot Edges (Road Segments) ---
    for edge in map_obj.edges:
        x_coords = [edge.node1.position.x, edge.node2.position.x]
        y_coords = [-edge.node1.position.y, -edge.node2.position.y] # Invert Y
        plt.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1)

    # --- Plot Road Names from the 'roads' list ---
    plotted_road_names = set() # Keep track of plotted names/locations
    if roads:
        for road in roads:
            try:
                start_pos = road["start"]
                end_pos = road["end"]
                road_name = road["name"]

                # Ensure start/end pos have x, y attributes (adjust if needed)
                if (hasattr(start_pos, 'x') and hasattr(start_pos, 'y') and
                    hasattr(end_pos, 'x') and hasattr(end_pos, 'y')):

                    x_coords = [start_pos.x, end_pos.x]
                    y_coords = [-start_pos.y, -end_pos.y] # Invert Y

                    # Calculate midpoint for text placement
                    mid_x = (x_coords[0] + x_coords[1]) / 2
                    mid_y = (y_coords[0] + y_coords[1]) / 2

                    # Create a key to prevent duplicate plotting in the same spot
                    position_tuple = (round(mid_x, 1), round(mid_y, 1))
                    name_pos_key = (road_name, position_tuple)

                    if name_pos_key not in plotted_road_names:
                        # Determine if the road is vertical
                        is_vertical = abs(x_coords[0] - x_coords[1]) < 1e-5
                        
                        rotation = 90 if is_vertical else 0
                        vertical_alignment = 'center' if is_vertical else 'bottom'
                        
                        plt.text(mid_x, mid_y, road_name,
                                 fontsize=7, color='darkgreen',
                                 ha='center', va=vertical_alignment, # Adjust VA for vertical roads
                                 rotation=rotation) # Apply rotation, Removed bbox
                        plotted_road_names.add(name_pos_key)
                else:
                     print(f"Warning: Skipping road '{road_name}' due to missing coordinate attributes.")

            except (KeyError, AttributeError) as e:
                print(f"Warning: Skipping a road due to missing data or incorrect format: {e}")
    # --- End Plot Road Names ---


    # --- Plot Nodes (Intersections, Supply Points) ---
    normal_nodes = []
    intersection_nodes = []
    supply_nodes = []

    for node in map_obj.nodes:
        if node.type == "normal":
            normal_nodes.append(node)
        elif node.type == "intersection":
            intersection_nodes.append(node)
        elif node.type == "supply":
            supply_nodes.append(node)

    # if normal_nodes:
    #     x_coords = [node.position.x for node in normal_nodes]
    #     y_coords = [node.position.y for node in normal_nodes]
    #     plt.scatter(x_coords, y_coords, c='blue', s=30, label='Normal Nodes')

    if intersection_nodes:
        x_coords = [node.position.x for node in intersection_nodes]
        y_coords = [-node.position.y for node in intersection_nodes]
        plt.scatter(x_coords, y_coords, c='red', s=50, label='Intersections')

    if supply_nodes:
        x_coords = [node.position.x for node in supply_nodes]
        y_coords = [-node.position.y for node in supply_nodes]
        plt.scatter(x_coords, y_coords, c='green', s=80, marker='*', label='Supply Points')
    
    for i, node in enumerate(spawning_locations):
        x_coords = node.x
        y_coords = -node.y
        plt.scatter(x_coords, y_coords, c='magenta', s=300, marker='o', label='spawning location')
        plt.text(x_coords, y_coords, str(i+1), fontsize=15, color='white',
                 ha='center', va='center', weight='bold')
    
    if destination:
        x_coords = destination.x
        y_coords = -destination.y
        plt.scatter(x_coords, y_coords, c='black', s=300, marker='o', label='destination')
        plt.text(x_coords, y_coords, 'D', fontsize=15, color='white',
                 ha='center', va='center', weight='bold')

    plt.title('Map with Multi-Agents Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_map_with_all_landmarks(map_obj: Map, landmarks: List[dict], roads: List[dict] = None, save_path: Optional[str] = None):
    plt.figure(figsize=(12, 8))

    # --- Plot Edges (Road Segments) ---
    for edge in map_obj.edges:
        x_coords = [edge.node1.position.x, edge.node2.position.x]
        y_coords = [-edge.node1.position.y, -edge.node2.position.y] # Invert Y
        plt.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1)

    # --- Plot Road Names from the 'roads' list ---
    plotted_road_names = set() # Keep track of plotted names/locations
    if roads:
        for road in roads:
            try:
                start_pos = road["start"]
                end_pos = road["end"]
                road_name = road["name"]

                # Ensure start/end pos have x, y attributes (adjust if needed)
                if (hasattr(start_pos, 'x') and hasattr(start_pos, 'y') and
                    hasattr(end_pos, 'x') and hasattr(end_pos, 'y')):

                    x_coords = [start_pos.x, end_pos.x]
                    y_coords = [-start_pos.y, -end_pos.y] # Invert Y

                    # Calculate midpoint for text placement
                    mid_x = (x_coords[0] + x_coords[1]) / 2
                    mid_y = (y_coords[0] + y_coords[1]) / 2

                    # Create a key to prevent duplicate plotting in the same spot
                    position_tuple = (round(mid_x, 1), round(mid_y, 1))
                    name_pos_key = (road_name, position_tuple)

                    if name_pos_key not in plotted_road_names:
                        # Determine if the road is vertical
                        is_vertical = abs(x_coords[0] - x_coords[1]) < 1e-5
                        
                        rotation = 90 if is_vertical else 0
                        vertical_alignment = 'center' if is_vertical else 'bottom'
                        
                        plt.text(mid_x, mid_y, road_name,
                                 fontsize=7, color='darkgreen',
                                 ha='center', va=vertical_alignment, # Adjust VA for vertical roads
                                 rotation=rotation) # Apply rotation, Removed bbox
                        plotted_road_names.add(name_pos_key)
                else:
                     print(f"Warning: Skipping road '{road_name}' due to missing coordinate attributes.")

            except (KeyError, AttributeError) as e:
                print(f"Warning: Skipping a road due to missing data or incorrect format: {e}")
    # --- End Plot Road Names ---


    # --- Plot Nodes (Intersections, Supply Points) ---
    normal_nodes = []
    intersection_nodes = []
    supply_nodes = []

    for node in map_obj.nodes:
        if node.type == "normal":
            normal_nodes.append(node)
        elif node.type == "intersection":
            intersection_nodes.append(node)
        elif node.type == "supply":
            supply_nodes.append(node)

    for i, landmark in enumerate(landmarks):
        x_coords = landmark['front_door_position'][0]
        y_coords = -landmark['front_door_position'][1]
        plt.scatter(x_coords, y_coords, c='magenta', s=40, marker='*', label='landmark')
        plt.text(x_coords, y_coords, str(i), fontsize=10, color='black',
                ha='right', va='bottom')
    
    # if normal_nodes:
    #     x_coords = [node.position.x for node in normal_nodes]
    #     y_coords = [node.position.y for node in normal_nodes]
    #     plt.scatter(x_coords, y_coords, c='blue', s=30, label='Normal Nodes')

    if intersection_nodes:
        x_coords = [node.position.x for node in intersection_nodes]
        y_coords = [-node.position.y for node in intersection_nodes]
        plt.scatter(x_coords, y_coords, c='red', s=50, label='Intersections')

    if supply_nodes:
        x_coords = [node.position.x for node in supply_nodes]
        y_coords = [-node.position.y for node in supply_nodes]
        plt.scatter(x_coords, y_coords, c='green', s=80, marker='*', label='Supply Points')

    plt.title('Map with Multi-Agents Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_map_multiagents_spawning_locations(map_obj: Map, spawning_locations: List[dict], roads: List[dict] = None, save_path: Optional[str] = None):
    plt.figure(figsize=(12, 8))

    # --- Plot Edges (Road Segments) ---
    for edge in map_obj.edges:
        x_coords = [edge.node1.position.x, edge.node2.position.x]
        y_coords = [-edge.node1.position.y, -edge.node2.position.y] # Invert Y
        plt.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1)

    # --- Plot Road Names from the 'roads' list ---
    plotted_road_names = set() # Keep track of plotted names/locations
    if roads:
        for road in roads:
            try:
                start_pos = road["start"]
                end_pos = road["end"]
                road_name = road["name"]

                # Ensure start/end pos have x, y attributes (adjust if needed)
                if (hasattr(start_pos, 'x') and hasattr(start_pos, 'y') and
                    hasattr(end_pos, 'x') and hasattr(end_pos, 'y')):

                    x_coords = [start_pos.x, end_pos.x]
                    y_coords = [-start_pos.y, -end_pos.y] # Invert Y

                    # Calculate midpoint for text placement
                    mid_x = (x_coords[0] + x_coords[1]) / 2
                    mid_y = (y_coords[0] + y_coords[1]) / 2

                    # Create a key to prevent duplicate plotting in the same spot
                    position_tuple = (round(mid_x, 1), round(mid_y, 1))
                    name_pos_key = (road_name, position_tuple)

                    if name_pos_key not in plotted_road_names:
                        # Determine if the road is vertical
                        is_vertical = abs(x_coords[0] - x_coords[1]) < 1e-5
                        
                        rotation = 90 if is_vertical else 0
                        vertical_alignment = 'center' if is_vertical else 'bottom'
                        
                        plt.text(mid_x, mid_y, road_name,
                                 fontsize=7, color='darkgreen',
                                 ha='center', va=vertical_alignment, # Adjust VA for vertical roads
                                 rotation=rotation) # Apply rotation, Removed bbox
                        plotted_road_names.add(name_pos_key)
                else:
                     print(f"Warning: Skipping road '{road_name}' due to missing coordinate attributes.")

            except (KeyError, AttributeError) as e:
                print(f"Warning: Skipping a road due to missing data or incorrect format: {e}")
    # --- End Plot Road Names ---


    # --- Plot Nodes (Intersections, Supply Points) ---
    normal_nodes = []
    intersection_nodes = []
    supply_nodes = []

    for node in map_obj.nodes:
        if node.type == "normal":
            normal_nodes.append(node)
        elif node.type == "intersection":
            intersection_nodes.append(node)
        elif node.type == "supply":
            supply_nodes.append(node)

    for i, spawning_location in enumerate(spawning_locations):
        x_coords = spawning_location[0]
        y_coords = -spawning_location[1]
        plt.scatter(x_coords, y_coords, c='magenta', s=40, marker='o', label='spawning location')
        plt.text(x_coords, y_coords, str(i+1), fontsize=10, color='black',
                ha='right', va='bottom')
    
    # if normal_nodes:
    #     x_coords = [node.position.x for node in normal_nodes]
    #     y_coords = [node.position.y for node in normal_nodes]
    #     plt.scatter(x_coords, y_coords, c='blue', s=30, label='Normal Nodes')

    if intersection_nodes:
        x_coords = [node.position.x for node in intersection_nodes]
        y_coords = [-node.position.y for node in intersection_nodes]
        plt.scatter(x_coords, y_coords, c='red', s=50, label='Intersections')

    if supply_nodes:
        x_coords = [node.position.x for node in supply_nodes]
        y_coords = [-node.position.y for node in supply_nodes]
        plt.scatter(x_coords, y_coords, c='green', s=80, marker='*', label='Supply Points')

    plt.title('Map with Multi-Agents Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()