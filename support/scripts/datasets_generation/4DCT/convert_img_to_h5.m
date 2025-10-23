% Extract 4DCT DIR-lab.com .img files, and convert them to h5 format.
% usage:
% 1)   Put the matlab .p files, that located in MatlabUtilityPack1 - v1.0
%       folder, with this script.
% 2)   Run the ImageImprt.p function in the matlab command window, with no
%        arguments.
% 3)    It will open a WinUI dialog, to selct the .img file.
% 4)    Once selected, it will open another WinUI importing options
%        window, do the following:
%       a)    Remove any sufixes from the variable name, like _s or -ssm (var
%              name is obtained from file name, and those in some cases has those
%              sufixes).
%       b)    Set the correct image dimentions, as mentioned in there
%              website https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html
%       c)    No other changes are needed, just press import.
% 5)   Do above steps, for all breathing phases, or images, within one case, so
%       there will be 10 matlab variables, contains each of the current case
%       pahses.
% 6)   Change script' case_id var to the case number, set target_dir to the
%       directory you want to output h5 files.
% 7)    Run the script, and it will dump all of the mentioned variables,
%        into h5 files, with naming caseID_phaseID.h5
% 8)    Clear out all matlab variables, and redo everything for the next
%        case.

case_id = 10;
case_id_str = int2str(case_id);
target_dir = join(['E:\System Folders\Downloads\Thesis\DIR 4DCT\h5\']);
key = '/ds';


   for phase_id = 0 : 9
       phase_id_str = int2str(phase_id);
       var_name = join(['case', case_id_str, '_T', phase_id_str, '0']);
       file_name = join([case_id_str, '_', phase_id_str, '.h5']);
       full_path = join([target_dir, file_name]);

       data = eval(var_name);

       h5create(full_path, key, size(data))
       h5write(full_path, key, data)
   end
