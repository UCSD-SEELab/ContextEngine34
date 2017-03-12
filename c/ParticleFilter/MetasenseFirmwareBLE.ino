#include "application.h" // needed for local compile

SYSTEM_MODE(SEMI_AUTOMATIC);
SYSTEM_THREAD(ENABLED);
STARTUP(System.enableFeature(FEATURE_RETAINED_MEMORY));
STARTUP(System.enableFeature(FEATURE_RESET_INFO));


#if (PLATFORM_ID == 6)
// Photon code here
PRODUCT_ID(790);
#elif (PLATFORM_ID == 10)
//Electron or other Particle device code here
PRODUCT_ID(2015);
#endif

PRODUCT_VERSION(6);

//#include <SD.h>
// Replaced by sd-card-library

//#include <avr/wdt.h>
//replaced by photon-wdgs but no solution for electron
//#include "photon-wdgs.h"

//#include <EEPROM.h> //Included in photon firmware
//#include <Time.h>
//#include <Wire.h> //Included in photon firmware
//#include <SPI.h> //Included in photon firmware

#include "Adafruit_ADS1015.h"
#include "PhotonConfig.h"
//#include "MirroringStream.h"
#include "SHT1x.h"
#include "Message.h"
/*#include "JsonDeserializer.h"
#include "JsonSerializer.h"*/
#include "AFE.h"
#include "VOC.h"
#include "CO2.h"
#include "Sensor.h"
#include "ServiceConnector.h"
#include "logger.h"
#include "PowerManager.h"

//Variables retined when in DEEP SLEEP
//retained unsigned long samplingInterval = 5000;
//retained long wifiStatus = -1;
retained bool usbMirror = false;
//retained StreamingType_t streamingType = streamAll;
retained unsigned long lastSetupTime = 0;
retained unsigned long lastReadingTime = 0;
retained unsigned long nextSyncTime = 0;
//retained bool sleepEnabled = true;
//retained bool vocInstalled = false;
//retained bool co2Installed = false;
retained bool init = true;
retained SensorEEPROMConfig_t SensorConfig;

retained adsGain_t currentGain = GAIN_TWOTHIRDS;
//retained int ResetSequenceLen = 0;

retained int BLE_KEY_PIN = D4;
retained int UNCONNECTED_CS_PIN = D6;

retained PowerEEPROMState_t PowerState;

PowerManager PM;

//Retained buffer of read data. We can use this to upload compressed batches
//to the cloud

retained unsigned char BinMsgBufferPos;
retained union BinMsgBuffer_t MsgBuffer;

// ----------------
bool temporarlyDisableSleep = false;
bool usbPassthrough = false;
int scheduledSetupCommand = 0;
bool wkupPinEnabled = false;

//Make sure that the sensor resets if the sensor is stuck in the loop for
//more than a minute
ApplicationWatchdog wd(WATCHDOG_TIMEOUT, System.reset);

int freeRam() {
	uint32_t freemem = System.freeMemory();
	return freemem;
}

Sensor sensor(HumSckPin, HumDataPin, BarCSPin, SDCSPin, UNCONNECTED_CS_PIN, ADS1115_ADDRESS_0, ADS1115_ADDRESS_1);

VOC voc(ADS1115_ADDRESS_0);
CO2 co2;

//MirroringStream mirroringStream(Serial1, &Serial, defaultMirrorToUSB);
ServiceConnector connector(/*mirroringStream,*/ sensor, voc, co2);


char buf[MAX_MSG_LEN+1];
// Cloud functions must return int and take one String
int processMsg(String extra) {
	INO_TRACE("Processing cloud command: %s\r\n", extra.c_str());
	//StringStream stream(extra);
	//InCmdMessage msg(stream, 100L);
	strncpy(buf, extra.c_str(), MAX_MSG_LEN);
	connector.receiveMessageWiFi(buf);
	//connector.processCommand(msg);
  return 0;
}

void button_clicked(system_event_t event, int param)
{
    int times = system_button_clicks(param);
    connector.setup_button_clicked(times);
}

void re_enable_sleep() {
	//Serial1.println("Reenable sleep");
	temporarlyDisableSleep = false;
}

Timer timer(30000, re_enable_sleep, true);

void setup()
{
	if (init)
	{
		usbMirror = false;
		lastSetupTime = 0;
		lastReadingTime = 0;
		nextSyncTime = 0;
		SensorConfig;
		currentGain = GAIN_TWOTHIRDS;
		//ResetSequenceLen = 0;

		if (BOARD_VERSION>=2.2) {
			BLE_KEY_PIN = D4;
			UNCONNECTED_CS_PIN = D6;
		} else {
			BLE_KEY_PIN = D6;
			UNCONNECTED_CS_PIN = D4;
		}

		BinMsgBufferPos = 0;

		PowerState.isAvail = false;
		PowerState.isCal = false;
		PowerState.charge = 0;
		PowerState.lastInit = 0;
		PowerState.lastCal = 0;
		PowerState.soc = 0;
		PowerState.canary = 0;
		PowerState.ccounter = 0;
	}

	//Manage changing state using setup button
	System.on(button_final_click, button_clicked);

	//System.on(reset_pending, reset_pending_event);

	//Cloud function to send commands via Cloud
	bool success = Particle.function("processMsg", processMsg);

	/*
	//To read the charge level
	pinMode(voltageLevelPin, INPUT);

	//Enable charging the battery when connected to USB
	pinMode(ChargeEnablePin, OUTPUT);
	digitalWrite(ChargeEnablePin, HIGH);*/

	//Configure seral ports
	Serial.begin(serialSpeed);		//USB uart on photon
	//Serial.blockOnOverrun(false); //To avoid blocking the sensor when logging
	 															//stuff and the serial is not available

	Serial1.begin(serialSpeed);		//Tx/Rx pins on photon
	Serial1.println();		//Tx/Rx pins on photon

	// -------------------------------------------------
	//This enable us to temprarly disable the sleep mode
	//when we reset the sensor using the reset button
	//We assume this is when we need to configure the sensor
	//and we want to be able to use the setup button for conifigs
	if (System.resetReason() == RESET_REASON_PIN_RESET) {
		temporarlyDisableSleep = true;
		INO_TRACE("Disable sleep for 30 secs");
		timer.start();
		//ResetSequenceLen = 0;
	}
	// -------------------------------------------------

	sensor.begin();
	connector.begin();

	if (SensorConfig.vocInstalled)
		voc.begin();
	if (SensorConfig.co2Installed)
		co2.begin();

	PM.begin(&PowerState);

	if (PM.isBatteryLow() && !PM.isChargingOrTrickling()){
		//TODO for debug remove in production
		PM.printPowerReport();
		//TODO end of debug stuff
		INO_TRACE("---------Low battery in begin. Go down for sleep.---------");
		//Sleep for 60 secs
		System.sleep(SLEEP_MODE_DEEP, 60000);
	}

	// If the reset was due to power issues
	if (System.resetReason() == RESET_REASON_POWER_MANAGEMENT ||
					 System.resetReason() == RESET_REASON_POWER_DOWN ||
					 System.resetReason() == RESET_REASON_POWER_BROWNOUT) {
		//TODO: Maybe we need to do something if battery died to reset power module??
		PM.printPowerReport();
		//TODO end of debug stuff
		if (System.resetReason() == RESET_REASON_POWER_MANAGEMENT)
			INO_TRACE("Recovered from reset with reason RESET_REASON_POWER_MANAGEMENT: %d\n", System.resetReason());
		if (System.resetReason() == RESET_REASON_POWER_DOWN)
			INO_TRACE("Recovered from reset with reason RESET_REASON_POWER_DOWN: %d\n", System.resetReason());
		if (System.resetReason() == RESET_REASON_POWER_BROWNOUT)
			INO_TRACE("Recovered from reset with reason RESET_REASON_POWER_BROWNOUT: %d\n", System.resetReason());
		//PM.reset();
	}

	bool wkupPinEnabled = false;


	init = false;
}

void loop()
{
	if (connector.updateReadings()){
		INO_TRACE("---------Update Readings returned true.---------\n");
		connector.processReadings();

		/*// PM.get* or PM.is* functions need to be called periodically for
		// SOC reading to stay accurate
		// i.e. one of these functions should be included in the loop: PM.isBatteryLow(), PM.getFuelLevel(), isCharging(), ...*/

		//PM.updateReadings must be called periodically
		PM.updateReadings();
		PM.printPowerReport();
		if (PM.isBatteryLow() && !PM.isChargingOrTrickling()){
			INO_TRACE("---------Low battery in loop. Go down for sleep.---------\n");
			//Sleep for 60 secs
			System.sleep(SLEEP_MODE_DEEP, 60000);
		}
	}
	connector.processCommands();
	//Make sure we diable the forced wakeup if the pin goes down
	sensor.initWakeupPinStatus();
}

char usb_msg[MAX_MSG_LEN+1];
int usb_pos=0;
void serialEvent()
{
	while (Serial.available()) {
		char c = Serial.read();
		if (usbPassthrough) {
			if (c == 4)
				usbPassthrough = false;
			else
				Serial1.write(c);
		} else {
			usb_msg[usb_pos++] = c;
			if (c=='\n')
			{
				usb_msg[usb_pos]=0;
				connector.receiveMessageUSB(usb_msg);
				usb_pos=0;
			}
			if (usb_pos>=MAX_MSG_LEN)
				usb_pos=0;
		}
	}
}
char ble_msg[MAX_MSG_LEN+1];
int ble_pos=0;
void serialEvent1()
{
	while(Serial1.available()) {
		char c = Serial1.read();
		if ((usbPassthrough || usbMirror) && sensor.isUSBSerialConnected())
			Serial.write(c);
		if(!usbPassthrough)
		{
			ble_msg[ble_pos++] = c;
			if (c=='\n')
			{
				ble_msg[ble_pos]=0;
				connector.receiveMessageBLE(ble_msg);
				ble_pos=0;
			}
			if (ble_pos>=MAX_MSG_LEN)
				ble_pos=0;
		}
	}
}
