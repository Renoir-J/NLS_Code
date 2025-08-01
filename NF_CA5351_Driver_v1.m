% Yueyun Chen
% 20250602 v1 basic controls over NF CA5351: zero check, change gain, change filter

% Notes

classdef NF_CA5351_Driver_v1 < VISA_IO_v1

    properties (SetAccess = protected)
        
        gain_list = round(10.^(3:10)); % list of gains in the order of ascending return values
        filter_time_list = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]; % list of filter time in the order of ascending return values

    end

    methods (Access = public)

        % constructor
        function obj = NF_CA5351_Driver_v1(addr,test)
            arguments
                addr {mustBeTextScalar} = "TCPIP0::10.97.108.250::5025::SOCKET";
                test {mustBeNumericOrLogical} = true; % test connection or not
            end
            obj = obj@VISA_IO_v1(addr,test)
        end
        
        % function to set gain value
        function set_gain(obj,gain)
            arguments
                obj 
                gain {mustBePositive,mustBeScalarOrEmpty} % gain value to set, 1e3 to 1e10
            end
            state = find(obj.gain_list == round(gain)); % look for match
            
            % check if input is valid
            if isempty(state)
                warning("Invalid gain value for NF CA5351!\nEntered value = %.1e\nAvailable values = %s\nGain is not changed.",gain,num2str(obj.gain_list,'%.0e, '))
                return
            end
            
            obj.writeline(":INPut:GAIN "+num2str(state)); % set gain value
        end

        % function to get gain value, WIP
        function gain_state = get_gain(obj)
            arguments
                obj 
            end
            gain_state = round(str2double(obj.writeread(":INPut:GAIN?"))); % get return value as an integer
            gain_state = obj.gain_list(gain_state); % get actual gain value
        end

        % function to set zero check
        function set_zerocheck(obj,state)
            arguments
                obj 
                state {mustBeMember(state,["ON" "OFF"])} 
            end
            obj.writeline(":INPut "+state)
        end

        % function to get zero check state
        function zc_state = get_zerocheck(obj)
            arguments
                obj 
            end
            zc_state = str2double(obj.writeread(":INPut?")); % 0 - OFF; 1 - ON
        end

        % function to set filter state
        function set_filter_state(obj,state)
            arguments
                obj 
                state {mustBeMember(state,["ON" "OFF"])} 
            end
            obj.writeline(":INPut:FILTer "+state)
        end

        % function to get filter state
        function filter_state = get_filter_state(obj)
            arguments
                obj 
            end
            filter_state = str2double(obj.writeread(":INPut:FILTer?")); % 0 - OFF; 1 - ON
        end

        % function to set filter time
        function set_filter_time(obj,time)
            arguments
                obj 
                time {mustBePositive,mustBeScalarOrEmpty} % filter time value to set, 1e-6 to 3e-1
            end
            state = find(obj.filter_time_list == time); % look for match
            
            % check if input is valid
            if isempty(state)
                warning("Invalid filter time value for NF CA5351!\nEntered value = %.1e\nAvailable values = %s\nFilter time is not changed.",time,num2str(obj.filter_time_list,'%.0e, '))
                return
            end
            
            obj.writeline(":INPut:FILTer:TIME "+num2str(state)); % set gain value
        end

        % function to get filter time
        function filter_time = get_filter_time(obj)
            arguments
                obj 
            end
            filter_time = round(str2double(obj.writeread(":INPut:FILTer:TIME?"))); % get return value as an integer
            filter_time = obj.filter_time_list(filter_time); % get actual filter time value
        end

        % function to set current suppression state
        function set_cs_state(obj,state)
            arguments
                obj 
                state {mustBeMember(state,["ON" "OFF"])} 
            end
            obj.writeline(":INPut:BIAS:CURRent:STATe "+state)
        end

        % function to get filter state
        function cs_state = get_cs_state(obj)
            arguments
                obj 
            end
            cs_state = str2double(obj.writeread(":INPut:BIAS:CURRent:STATe?")); % 0 - OFF; 1 - ON
        end

        % reset instrument
        function reset(obj)
            arguments
                obj 
            end
            obj.writeline("*RST");
        end

    end

end