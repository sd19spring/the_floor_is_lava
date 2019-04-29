let muteDict = {};  // for storing whether the camera is turned on or off; needs to be globally accessible

function createDivs(numCaps) {
    // Dynamically creates the html elements used to display the camera feeds - numCaps defines how many cameras there
    // are to display.
    let i;  // for iteration
    for (i = numCaps; i >= 0; i--) {
        // Create a new, plain <span> element
        var frame = document.createElement("div");
        var eye_str = '/' + i;
        frame.setAttribute('class', 'capture_div');
        frame.innerHTML = "<img alt='Camera frame' src=" + eye_str + " class='capture'/>" +
            "<div class='row'><div onclick='muteCam(" + i + ")' id='capToggle" + i + "' " +
            "class='cover button_small forty' " + ">Mute</div>" + "<div class='cover button_small forty' " +
            "id='calibrateToggle" + i + "' onclick='calibCam(" + i + ")'>Calibrate</div></div>";
        // Get a reference to the element, before we want to insert the element
        var sp2 = document.getElementById("childElement");
        // Get a reference to the parent element
        var parentDiv = sp2.parentNode;
        // Insert the new element into the DOM before sp2
        parentDiv.insertBefore(frame, sp2.nextSibling);

        muteDict[i] = 1;  // 1 indicates that the camera should be turned on; 0 indicates that it shouldn't
        calibDict[i] = 1;  // 1 indicates that the camera is not using calibration; 0 indicates that it is
    }
}

// these need to be globally accessible
let switchBool = '';
let calibDict = {};
let firstTimeFlag = false;  // boolean for whether the record button has been pressed yet

function changeSwitch(url) {
    // ajax the JSON to the server
    if (switchBool === 'true') {
        switchBool = 'false';
    } else {
        switchBool = 'true';
    }
    if (switchBool === true && firstTimeFlag === true) {
        document.getElementById('result').style.visibility = "visible";
    }
    firstTimeFlag = true;
    $.post(url, switchBool, changeSwitchButton(switchBool));
    // stop link reloading the page
    event.preventDefault();
}

function muteCam(capNum) {
    if (muteDict[capNum] === 1) {
        // turn the camera off
        $.post("/cap_switch", {capNum: capNum, record: 0}, changeButtonCam(capNum, muteDict[capNum], 'Unmute',
            'capToggle'));
        // stop link reloading the page
        event.preventDefault();
        muteDict[capNum] = 0;
    } else {
        // turn the camera on
        $.post("/cap_switch", {capNum: capNum, record: 1}, changeButtonCam(capNum, muteDict[capNum], 'Mute',
            'capToggle'));
        // stop link reloading the page
        event.preventDefault();
        muteDict[capNum] = 1;
    }
}

function calibCam(capNum) {
    if (calibDict[capNum] === 1) {
        // turn initialize
        $.post("/calib_switch", {capNum: capNum, record: 0}, changeButtonCam(capNum, calibDict[capNum], 'Revert',
            'calibrateToggle'));
        // stop link reloading the page
        event.preventDefault();
        calibDict[capNum] = 0;
    } else {
        // turn the camera on
        $.post("/calib_switch", {capNum: capNum, record: 1}, changeButtonCam(capNum, calibDict[capNum],
            'Calibrate', 'calibrateToggle'));
        // stop link reloading the page
        event.preventDefault();
        calibDict[capNum] = 1;
    }
}


function changeSwitchButton(switchBool) {
    // Used to change the appearance and properties of the button used to toggle the recording of the heatmap.
    // switchBool is a global integer boolean that keeps track of the current state of the button.
    if (switchBool === 'true') {
        document.getElementById("switch").innerHTML = "<a>Stop Recording</a>";
        document.getElementById("switch").className = "cover button_red";
    } else {
        document.getElementById("switch").innerHTML = "<a>Create Heatmap ➡️</a>";
        document.getElementById("switch").className = "cover button";
    }
}

function changeButtonCam(capNum, switchBoolCam, text, elID) {
    // Used to change the appearance and properties of the button used to toggle properties of the cameras. capNum is
    // the index of the camera capture, switchBoolCam is an integer boolean (given by muteDict[capNum]) that determines
    // what sate the button should be in. Text is the text that the button should be displaying, and elID is the element
    // id of the button that should be changed. buttonSmall is a boolean value that determines whether the class name
    // of the div should include button_red_small and button_small or just button_red and button
    var elementID = '';
    elementID = elID + capNum;
    if (switchBoolCam === 1) {
        document.getElementById(elementID).innerHTML = "<a>" + text + "</a>";
        document.getElementById(elementID).className = "cover forty button_red_small";
    } else {
        document.getElementById(elementID).innerHTML = "<a>" + text + "</a>";
        document.getElementById(elementID).className = "cover forty button_small";
    }
}