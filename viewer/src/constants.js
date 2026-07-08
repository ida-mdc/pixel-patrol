export const DEFAULT_PALETTE     = 'tab10';
export const GROUP_ALL           = 'all';
export const GROUP_COL_ALIAS     = '__group__';
export const FILE_ROW_NUMBER     = 'file_row_number';
export const WIDGET_CONTAINER_ID = 'widget-container';

// Timestamp columns, and the SQL STRFTIME format used to render them.
export const DATE_COLS = new Set(['modification_date']);
export const DATE_FMT  = "'%Y-%m-%d %H:%M:%S'";

// DOM element IDs shared across modules
export const ID_WELCOME_SCREEN   = 'welcome-screen';
export const ID_MAIN_APP         = 'main-app';
export const ID_LOADING_OVERLAY  = 'loading-overlay';
export const ID_SIDEBAR_BACKDROP = 'sidebar-backdrop';