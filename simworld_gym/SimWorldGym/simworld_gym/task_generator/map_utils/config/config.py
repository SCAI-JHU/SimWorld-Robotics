class Config:
    SEED = 42
    DT = 1
    UE_UPDATE_DT = 0.2

    SIDEWALK_OFFSET = 1700  # distance between the sidewalk and the road center
    NUM_THREADS = 20

    # DELIVERY
    MAX_ORDERS = 3
    PROBABILITY_TO_CREATE_NEW_ORDER = 0.8
    COST_OF_BEVERAGE = 5
    DELIVERY_MAN_RECOVER_ENERGY_AMOUNT = 60
    DELIVERY_MAN_MEET_DISTANCE = 500
    PRICE_OF_BIKE = 100
    DIFFICULTY = 'easy'  # easy, medium, hard

    # DELIVERY MAN
    DELIVERY_MAN_WALK_ARRIVE_WAYPOINT_DISTANCE = 200
    DELIVERY_MAN_DRIVE_ARRIVE_WAYPOINT_DISTANCE = 400
    DELIVERY_MAN_INITIAL_ENERGY = 100
    DELIVERY_MAN_MIN_SPEED = 100  # unit: cm/s
    DELIVERY_MAN_MAX_SPEED = 250  # unit: cm/s
    USE_PLANNER = False

    # Navigation
    PID_KP = 0.15
    PID_KI = 0.005
    PID_KD = 0.12

    # UE
    DELIVERY_MAN_MODEL_PATH = "/Game/TrafficSystem/Pedestrian/BP_DeliveryMan.BP_DeliveryMan_C"
    DELIVERY_MANAGER_MODEL_PATH = "/Game/TrafficSystem/DeliveryManager.DeliveryManager_C"
    SCOOTER_MODEL_PATH = "/Game/ScooterAssets/Blueprints/BP_Scooter_Pawn.BP_Scooter_Pawn_C"
