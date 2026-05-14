SCENE_TASK_PROMPT = {
    "1ext": "Open the cabinet.",
    "3a": "Open the door.",
    "3b": "Open the door.",
    "3c": "Open the door.",
    "3d": "Open the door.",
    "5a": "Open the bottle.",
    "5b": "Open the bottle.",
    "5c": "Open the bottle.",
    "5d": "Close the bottle.",
    "5e": "Close the bottle.",
    "5f": "Close the bottle.",
    "5g": "Open the bottle.",
    "5h": "Close the bottle cap.",
}

CABINET_TEST_USD = ['45677','46130'] # '19179','45135','31249','34178','45194',
DOOR_TEST_USD = ['99689669960001','99689669962036','99692809960003'] #'99660089960018','99660089962003','99660099962042',
BOTTLE_TEST_USD = ['b4','b17','3b','8b','14b','b3','b8'] ##'b4','b17','3b','8b','14b','b3','b8'


SCENE_TASK_PROMPT_GUIDE = {
    "1ext": "Open the cabinet following the guide.",
    "3a": "Open the door following the guide.",
    "3b": "Open the door following the guide.",
    "3c": "Open the door following the guide.",
    "3d": "Open the door following the guide.",
    "5a": "Open the bottle following the guide.",
    "5b": "Open the bottle following the guide.",
    "5c": "Open the bottle following the guide.",
    "5d": "Close the bottle following the guide.",
    "5e": "Close the bottle following the guide.",
    "5f": "Close the bottle following the guide.",
    "5g": "Open the bottle following the guide.",
    "5h": "Close the bottle cap following the guide.",
}


SCENE_TASK_PROMPT_INSTRUCTION = {
    "1ext": "Find the arrow guide and open the indicated drawer.",
    "3a": "Open the door, rotate clockwise and push.",
    "3b": "Open the door, rotate counter-clockwise and push.",
    "3c": "Open the door, rotate clockwise and pull.",
    "3d": "Open the door, rotate counter-clockwise and pull.",
    "5a": "Grip the cap on the sides indicated by the 'squeeze' arrow and open the bottle in counter-clockwise direction.",
    "5b": "Open the bottle in counter-clockwise direction.",
    "5c": "Open the bottle in clockwise direction.",
    "5d": "Close the bottle in clockwise direction.",
    "5e": "Close the bottle in counter-clockwise direction.",
    "5f": "Close the bottle in clockwise direction.",
    "5g": "Grip the cap on the sides indicated by the 'squeeze' arrow and open the bottle in clockwise direction.",
    "5h": "Close the bottle in counter-clockwise direction.",
}


SCENE_TASK_PROMPT_SEM = {
    "1ext": "Find the arrow guide and open the indicated drawer.",
    "3a": "Open the door following the text or symbolic guide.",
    "3b": "Open the door following the text or symbolic guide.",
    "3c": "Open the door following the text or symbolic guide.",
    "3d": "Open the door following the text or symbolic guide.",
    "5a": "Open the bottle following the text or symbolic guide.",
    "5b": "Open the bottle following the text or symbolic guide.",
    "5c": "Open the bottle following the text or symbolic guide.",
    "5d": "Close the bottle following the text or symbolic guide.",
    "5e": "Close the bottle following the text or symbolic guide.",
    "5f": "Close the bottle following the text or symbolic guide.",
    "5g": "Open the bottle following the text or symbolic guide.",
    "5h": "Close the bottle cap following the text or symbolic guide.",
}

PROMPT_REAL_G = {
    "Open the left drawer following guide": "Open the cabinet following the guide.",
    "Open the drawer following guide": "Open the cabinet following the guide.",
    "open bottle cw": "Open the bottle following the guide.",
    "open bottle ccw": "Open the bottle following the guide.",
    "3b": "Open the door following the guide.",
    "3c": "Open the door following the guide.",
    "3d": "Open the door following the guide.",
    "5a": "Open the bottle following the guide.",
    "5b": "Open the bottle following the guide.",
    "5c": "Open the bottle following the guide.",
    "5d": "Close the bottle following the guide.",
    "5e": "Close the bottle following the guide.",
    "5f": "Close the bottle following the guide.",
    "5g": "Open the bottle following the guide.",
    "5h": "Close the bottle cap following the guide.",
}

PROMPT_REAL_LI = {
    "Open the left drawer following guide": "Find the arrow guide and open the indicated drawer. The indicated drawer is left drawer. ",
    "Open the drawer following guide": "Find the arrow guide and open the indicated drawer. The indicated drawer is right drawer.",
    "open bottle cw": "Open the bottle in clockwise direction.",
    "open bottle ccw": "Open the bottle in counter-clockwise direction.",
    "3b": "Open the door following the guide.",
    "3c": "Open the door following the guide.",
    "3d": "Open the door following the guide.",
    "5a": "Open the bottle following the guide.",
    "5b": "Open the bottle following the guide.",
    "5c": "Open the bottle following the guide.",
    "5d": "Close the bottle following the guide.",
    "5e": "Close the bottle following the guide.",
    "5f": "Close the bottle following the guide.",
    "5g": "Open the bottle following the guide.",
    "5h": "Close the bottle cap following the guide.",
}