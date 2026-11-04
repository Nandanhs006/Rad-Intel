/*
 * Rad-Intel's Google Drive backup gateway.
 *
 * Paste this file into a standalone Apps Script project owned by the same
 * Google account that owns the private reviewed-radiographs folder. Deploy it
 * as a Web app that executes as you and is accessible to anyone. Access is
 * still restricted: every request must have a fresh HMAC signature created by
 * the Rad-Intel server.
 *
 * Project Settings -> Script properties (not this source file):
 *   RAD_INTEL_DRIVE_FOLDER_ID = the private Drive folder id
 *   RAD_INTEL_GATEWAY_SECRET  = the same random secret set on the server
 */

const SECRET_PROPERTY = 'RAD_INTEL_GATEWAY_SECRET';
const FOLDER_PROPERTY = 'RAD_INTEL_DRIVE_FOLDER_ID';
const MAX_AGE_SECONDS = 300;

function doGet() {
  return json_({ok: true, service: 'Rad-Intel Drive backup'});
}

function doPost(event) {
  try {
    if (!event || !event.postData || !event.postData.contents) {
      throw new Error('A JSON request body is required.');
    }
    const request = JSON.parse(event.postData.contents);
    verify_(request);
    if (request.action === 'status') return status_();
    if (request.action === 'upload') return upload_(request);
    if (request.action === 'download') return download_(request);
    if (request.action === 'delete') return remove_(request);
    throw new Error('Unsupported backup action.');
  } catch (error) {
    // Do not send stack traces or property values to a remote caller.
    return json_({ok: false, error: String(error && error.message || 'Backup request failed.')});
  }
}

function status_() {
  const folder = folder_();
  return json_({ok: true, folder_id: folder.getId(), folder_name: folder.getName()});
}

function upload_(request) {
  requireString_(request.name, 'File name');
  requireString_(request.mime, 'File type');
  requireString_(request.data_b64, 'File data');
  if (!/^[a-f0-9]{64}\.(jpg|png)$/.test(request.name)) {
    throw new Error('Unexpected backup file name.');
  }
  if (request.mime !== 'image/jpeg' && request.mime !== 'image/png') {
    throw new Error('Unexpected backup file type.');
  }
  const bytes = Utilities.base64Decode(request.data_b64);
  const file = folder_().createFile(Utilities.newBlob(bytes, request.mime, request.name));
  return json_({ok: true, file_id: file.getId()});
}

function download_(request) {
  requireString_(request.file_id, 'File id');
  try {
    const file = backupFile_(request.file_id);
    return json_({ok: true, found: true, data_b64: Utilities.base64Encode(file.getBlob().getBytes())});
  } catch (error) {
    // A deleted or unknown Drive id has no useful recovery path here.
    return json_({ok: true, found: false});
  }
}

function remove_(request) {
  requireString_(request.file_id, 'File id');
  try {
    // This removes it from the backup folder and moves it to the owner's bin.
    backupFile_(request.file_id).setTrashed(true);
  } catch (error) {
    return json_({ok: true, deleted: false});
  }
  return json_({ok: true, deleted: true});
}

function folder_() {
  const id = PropertiesService.getScriptProperties().getProperty(FOLDER_PROPERTY);
  if (!id) throw new Error('The backup folder is not configured in Script properties.');
  return DriveApp.getFolderById(id);
}

function backupFile_(id) {
  const file = DriveApp.getFileById(id);
  const expectedFolderId = folder_().getId();
  const parents = file.getParents();
  while (parents.hasNext()) {
    if (parents.next().getId() === expectedFolderId) return file;
  }
  throw new Error('The requested file is not in the configured backup folder.');
}

function verify_(request) {
  if (!request || typeof request !== 'object') throw new Error('Invalid request.');
  requireString_(request.action, 'Action');
  requireString_(request.nonce, 'Nonce');
  requireString_(request.signature, 'Signature');
  if (!Number.isInteger(request.timestamp) ||
      Math.abs(Math.floor(Date.now() / 1000) - request.timestamp) > MAX_AGE_SECONDS) {
    throw new Error('Expired backup request.');
  }
  const secret = PropertiesService.getScriptProperties().getProperty(SECRET_PROPERTY);
  if (!secret) throw new Error('The backup secret is not configured in Script properties.');
  const copy = {};
  Object.keys(request).forEach((key) => { if (key !== 'signature') copy[key] = request[key]; });
  const expected = Utilities.base64Encode(
    Utilities.computeHmacSha256Signature(canonical_(copy), secret)
  );
  if (!sameText_(expected, request.signature)) throw new Error('Invalid backup signature.');
  const cache = CacheService.getScriptCache();
  const nonceKey = 'radintel-nonce-' + Utilities.base64Encode(
    Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, request.nonce)
  );
  if (cache.get(nonceKey)) throw new Error('Duplicate backup request.');
  cache.put(nonceKey, '1', MAX_AGE_SECONDS);
}

function canonical_(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(canonical_).join(',') + ']';
  return '{' + Object.keys(value).sort().map((key) => JSON.stringify(key) + ':' +
    canonical_(value[key])).join(',') + '}';
}

function sameText_(left, right) {
  if (left.length !== right.length) return false;
  let different = 0;
  for (let i = 0; i < left.length; i++) different |= left.charCodeAt(i) ^ right.charCodeAt(i);
  return different === 0;
}

function requireString_(value, name) {
  if (typeof value !== 'string' || !value) throw new Error(name + ' is required.');
}

function json_(value) {
  return ContentService.createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}
